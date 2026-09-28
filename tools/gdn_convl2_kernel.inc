// [gdn-convl2-begin]  (sed-extracted by tools/convl2_proto.cu — keep self-contained)
// --- batched conv+silu: out[t][ch] = silu(sum_j cw[ch][j]*hist_j + cw[ch][3]*x[t])
// hist_j = x[t-3+j], t-3+j < 0 reads convst (recent 3 pre-prefill inputs, oldest
// first — matches k_gdn_conv). Not in-place: out must differ from x.
// float4 over channels (qkv % 4 == 0): per-lane op order identical to the
// scalar version -> bit-exact, 4x fewer threads.
__global__ void k_gdn_conv_b(const float* __restrict__ x, const float* __restrict__ cw,
                             const float* __restrict__ convst, float* __restrict__ out,
                             int P, int qkv) {
  int64_t i = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) * 4;
  if (i >= (int64_t)P * qkv) return;
  int t = (int)(i / qkv), ch = (int)(i - (int64_t)t * qkv);
  float4 xv = *(const float4*)(x + i);
  float4 a;
  a.x = cw[(ch + 0) * 4 + 3] * xv.x;
  a.y = cw[(ch + 1) * 4 + 3] * xv.y;
  a.z = cw[(ch + 2) * 4 + 3] * xv.z;
  a.w = cw[(ch + 3) * 4 + 3] * xv.w;
#pragma unroll
  for (int j = 0; j < 3; j++) {
    int tt = t - 3 + j;
    const float* src =
        tt >= 0 ? x + (size_t)tt * qkv + ch : convst + (tt + 3) * qkv + ch;
    float4 v = *(const float4*)src;
    a.x += cw[(ch + 0) * 4 + j] * v.x;
    a.y += cw[(ch + 1) * 4 + j] * v.y;
    a.z += cw[(ch + 2) * 4 + j] * v.z;
    a.w += cw[(ch + 3) * 4 + j] * v.w;
  }
  float4 o;
  o.x = a.x / (1.f + expf(-a.x));
  o.y = a.y / (1.f + expf(-a.y));
  o.z = a.z / (1.f + expf(-a.z));
  o.w = a.w / (1.f + expf(-a.w));
  *(float4*)(out + i) = o;
}

// --- write back conv state: convst[j] = last-3 inputs after P tokens.
// Per-channel independent; old values are read into registers before writing.
__global__ void k_convst_update(float* __restrict__ convst, const float* __restrict__ x,
                                int P, int qkv) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= qkv) return;
  float o1 = convst[qkv + ch], o2 = convst[2 * qkv + ch];
  float n2 = x[(size_t)(P - 1) * qkv + ch];
  float n1 = P >= 2 ? x[(size_t)(P - 2) * qkv + ch] : o2;
  // P=1 keeps old rows 1,2; P=2 keeps only old row 2.
  float n0 = P >= 3 ? x[(size_t)(P - 3) * qkv + ch] : (P == 2 ? o2 : o1);
  convst[ch] = n0;
  convst[qkv + ch] = n1;
  convst[2 * qkv + ch] = n2;
}

// --- batched GDN q/k L2 norm + q scale: grid (hk, P) over [P, qkvstride] rows.
// q heads at t*qkvstride + h*dk, k heads at t*qkvstride + hk*dk + h*dk.
__global__ void k_l2norm_qk_b(float* __restrict__ qkv, int dk, float qscale,
                              int qkvstride, int hk) {
  __shared__ float sm[128];
  int h = blockIdx.x;
  int t = threadIdx.x;
  float* qh = qkv + (size_t)blockIdx.y * qkvstride + h * dk;
  float* kh = qkv + (size_t)blockIdx.y * qkvstride + (size_t)hk * dk + h * dk;
  float v = (t < dk) ? qh[t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float iq = rsqrtf(sm[0] + 1e-6f);
  __syncthreads();
  float kv = (t < dk) ? kh[t] : 0.f;
  sm[t] = kv * kv;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float ik = rsqrtf(sm[0] + 1e-6f);
  if (t < dk) {
    qh[t] = v * iq * qscale;
    kh[t] = kv * ik;
  }
}

// --- fused k_gdn_conv_b + k_l2norm_qk_b (dk == 128 only). One wave32 per
// 128-channel slice of one token, float4 per lane; slices [0, nqk) are q heads
// (l2norm + qscale), [nqk, 2*nqk) k heads (l2norm), the rest v (conv only).
// Bit-identical to the pair: the conv expressions are copied verbatim (same
// fp-contraction), and the l2norm LDS tree (off = 64,32,16,8,4,2,1 over
// elements) is replayed exactly — element 4*lane+c, so off 64..4 are lane
// offsets 16..1 per component c and off 2,1 end in lane 0 as
// (s0+s2)+(s1+s3). fp contract is off in the reduction scope so v*v is not
// fused into the first add (the unfused kernel squares into LDS first).
// Saves the l2norm read+write of the q/k columns (~268 MB/layer at P=8192).
// grid (ceil(qkv/1024), P), 256 threads.
__global__ void __launch_bounds__(256)
    k_gdn_conv_l2n_b(const float* __restrict__ x, const float* __restrict__ cw,
                     const float* __restrict__ convst, float* __restrict__ out,
                     int P, int qkv, int nqk, float qscale) {
  const int lane = threadIdx.x & 31;
  const int slice = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (slice * 128 >= qkv) return;  // wave-uniform
  const int t = blockIdx.y, ch = slice * 128 + lane * 4;
  const int64_t i = (int64_t)t * qkv + ch;
  float4 xv = *(const float4*)(x + i);
  float4 a;
  a.x = cw[(ch + 0) * 4 + 3] * xv.x;
  a.y = cw[(ch + 1) * 4 + 3] * xv.y;
  a.z = cw[(ch + 2) * 4 + 3] * xv.z;
  a.w = cw[(ch + 3) * 4 + 3] * xv.w;
#pragma unroll
  for (int j = 0; j < 3; j++) {
    int tt = t - 3 + j;
    const float* src =
        tt >= 0 ? x + (size_t)tt * qkv + ch : convst + (tt + 3) * qkv + ch;
    float4 v = *(const float4*)src;
    a.x += cw[(ch + 0) * 4 + j] * v.x;
    a.y += cw[(ch + 1) * 4 + j] * v.y;
    a.z += cw[(ch + 2) * 4 + j] * v.z;
    a.w += cw[(ch + 3) * 4 + j] * v.w;
  }
  float4 o;
  o.x = a.x / (1.f + expf(-a.x));
  o.y = a.y / (1.f + expf(-a.y));
  o.z = a.z / (1.f + expf(-a.z));
  o.w = a.w / (1.f + expf(-a.w));
  if (slice < 2 * nqk) {  // wave-uniform
    // contract(off): HIP's __fmul_rn/__fadd_rn are plain * / + on amdgcn and
    // would still fuse the first square into fma(x, x, shfl) (measured).
#pragma clang fp contract(off)
    float s0 = o.x * o.x, s1 = o.y * o.y;
    float s2 = o.z * o.z, s3 = o.w * o.w;
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) {
      s0 = s0 + __shfl_down(s0, d, 32);
      s1 = s1 + __shfl_down(s1, d, 32);
      s2 = s2 + __shfl_down(s2, d, 32);
      s3 = s3 + __shfl_down(s3, d, 32);
    }
    float tot = (s0 + s2) + (s1 + s3);
    tot = __shfl(tot, 0, 32);
    const float inv = rsqrtf(tot + 1e-6f);
    if (slice < nqk) {
      o.x = o.x * inv * qscale;
      o.y = o.y * inv * qscale;
      o.z = o.z * inv * qscale;
      o.w = o.w * inv * qscale;
    } else {
      o.x = o.x * inv;
      o.y = o.y * inv;
      o.z = o.z * inv;
      o.w = o.w * inv;
    }
  }
  *(float4*)(out + i) = o;
}
// [gdn-convl2-end]
