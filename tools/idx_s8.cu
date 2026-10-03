// idx_s8.cu — k_index_scores_t64 的短 staging 变体（ST=8/4）A/B：
// 逐 bit 对比 + 计时。动机：t64 LDS 25KB → 每 CU 只能 2 CTA（VGPR 允许 4）；
// staging 深度减半到 8 后 LDS 12.5KB → 4 CTA。每个 (q,h,k) 的 FMA 链
// d 顺序不变（只是 stage 切细），输出应逐 bit 相同。
// Build: hipcc -O3 -I src/gpu --offload-arch=gfx1151 -o /tmp/idx_s8 tools/idx_s8.cu
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "parts/09_kernels_index.inc"

#define CK(x)                                                                    \
  do {                                                                           \
    hipError_t e_ = (x);                                                         \
    if (e_ != hipSuccess) {                                                      \
      fprintf(stderr, "hip error %d (%s) at %d\n", (int)e_,                      \
              hipGetErrorString(e_), __LINE__);                                  \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

// ST=8 staging：每 stage 只装 8 个 d（q 行 2×float4，k 行 1×float4）。
// d0 序列 (s*8 + par*64) & 127, s=0..15 覆盖全部 128 d，逐元素 FMA 顺序与
// t64 完全一致。LDS: qs 8x260x4 = 8320 B, ks 8x132x4 = 4224 B。
__global__ void __launch_bounds__(256)
k_index_scores_s8(const float* __restrict__ q, const float* __restrict__ keys,
                  float* __restrict__ out, int nb, int count, int first) {
  const int par = (int)blockIdx.x & 1;
  const int S = ((int)blockIdx.x >> 1) * 256 + par * 32;
  const int q0 = (int)blockIdx.y * 64;
  if (q0 >= count || S >= min(nb, (first + min(q0 + 64, count)) / 4)) return;
  __shared__ __align__(16) float qs[8][260];   // [d][query*4 + head]
  __shared__ __align__(16) float ks[8][132];   // [d][tile key]
  const int tid = threadIdx.x;
  const int qg = tid >> 4, kg = tid & 15;
  const float* kp;
  {
    const int j = tid >> 1, key = S + 64 * (j >> 5) + (j & 31);
    kp = key < nb ? keys + (size_t)key * 128 + (tid & 1) * 4 : nullptr;
  }
  float4 pq[2], pk;
  auto load = [&](int d0) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int j = tid + 256 * i, row = j >> 1, dd = (j & 1) * 4;
      pq[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (q0 + (row >> 2) < count)
        pq[i] = *reinterpret_cast<const float4*>(q + (size_t)(q0 * 4 + row) * 128 + d0 + dd);
    }
    pk = make_float4(0.f, 0.f, 0.f, 0.f);
    if (kp) pk = *reinterpret_cast<const float4*>(kp + d0);
  };
  float acc[4][4][8];  // [query][head][key]
#pragma unroll
  for (int a = 0; a < 4; a++)
#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
      for (int c = 0; c < 8; c++) acc[a][h][c] = 0.f;
  load((par * 64) & 127);
#pragma unroll 1
  for (int s = 0; s < 16; s++) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int j = tid + 256 * i, row = j >> 1, dd = (j & 1) * 4;
      qs[dd][row] = pq[i].x; qs[dd + 1][row] = pq[i].y;
      qs[dd + 2][row] = pq[i].z; qs[dd + 3][row] = pq[i].w;
    }
    {
      const int j = tid >> 1, dd = (tid & 1) * 4;
      ks[dd][j] = pk.x; ks[dd + 1][j] = pk.y;
      ks[dd + 2][j] = pk.z; ks[dd + 3][j] = pk.w;
    }
    __syncthreads();
    if (s < 15) load(((s + 1) * 8 + par * 64) & 127);
#pragma unroll
    for (int d = 0; d < 8; d++) {
      const float4 a0 = *reinterpret_cast<const float4*>(&qs[d][qg * 16]);
      const float4 a1 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 4]);
      const float4 a2 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 8]);
      const float4 a3 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 12]);
      const float4 k0 = *reinterpret_cast<const float4*>(&ks[d][kg * 4]);
      const float4 k1 = *reinterpret_cast<const float4*>(&ks[d][64 + kg * 4]);
      const float qv[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                            a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
      const float kk[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll
      for (int a = 0; a < 4; a++)
#pragma unroll
        for (int h = 0; h < 4; h++)
#pragma unroll
          for (int c = 0; c < 8; c++)
            acc[a][h][c] = fmaf(qv[a * 4 + h], kk[c], acc[a][h][c]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int a = 0; a < 4; a++) {
    const int t = q0 + qg * 4 + a;
    if (t >= count) continue;
    const int lim = min(nb, (first + t + 1) / 4);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const int j = kg * 4 + (c & 3) + 64 * (c >> 2);
      const int b = S + 64 * (j >> 5) + (j & 31);
      if (b < lim) {
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < 4; h++) s += fmaxf(0.f, acc[a][h][c]);
        out[(size_t)t * nb + b] = s * 0.08838834764831845f;
      }
    }
  }
}

// ST=4 变体：LDS qs 4x260 + ks 4x132 = 6.3 KB（受 VGPR 191 限仍 4 CTA，
// barrier 更密，看个趋势）。
__global__ void __launch_bounds__(256)
k_index_scores_s4(const float* __restrict__ q, const float* __restrict__ keys,
                  float* __restrict__ out, int nb, int count, int first) {
  const int par = (int)blockIdx.x & 1;
  const int S = ((int)blockIdx.x >> 1) * 256 + par * 32;
  const int q0 = (int)blockIdx.y * 64;
  if (q0 >= count || S >= min(nb, (first + min(q0 + 64, count)) / 4)) return;
  __shared__ __align__(16) float qs[4][260];
  __shared__ __align__(16) float ks[4][132];
  const int tid = threadIdx.x;
  const int qg = tid >> 4, kg = tid & 15;
  // k: 128 keys x 4 d = 128 float4；仅 tid<128 参与装载（每 key 4 d）
  const int kkey = S + 64 * ((tid & 127) >> 5) + (tid & 31);
  const float* kpp = (tid < 128 && kkey < nb) ? keys + (size_t)kkey * 128 : nullptr;
  float4 pq, pk;
  auto load = [&](int d0) {
    // q: 256 rows x 4 d = 256 float4 -> 1 per thread
    const int row = tid, dd = 0;  // 256 threads, 256 rows
    pq = make_float4(0.f, 0.f, 0.f, 0.f);
    if (q0 + (row >> 2) < count)
      pq = *reinterpret_cast<const float4*>(q + (size_t)(q0 * 4 + row) * 128 + d0 + dd);
    pk = make_float4(0.f, 0.f, 0.f, 0.f);
    if (kpp) pk = *reinterpret_cast<const float4*>(kpp + d0);
  };
  float acc[4][4][8];
#pragma unroll
  for (int a = 0; a < 4; a++)
#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
      for (int c = 0; c < 8; c++) acc[a][h][c] = 0.f;
  load((par * 64) & 127);
#pragma unroll 1
  for (int s = 0; s < 32; s++) {
    {
      const int row = tid;
      qs[0][row] = pq.x; qs[1][row] = pq.y; qs[2][row] = pq.z; qs[3][row] = pq.w;
    }
    if (tid < 128) {
      const int j = tid;
      ks[0][j] = pk.x; ks[1][j] = pk.y;
      ks[2][j] = pk.z; ks[3][j] = pk.w;
    }
    __syncthreads();
    if (s < 31) load(((s + 1) * 4 + par * 64) & 127);
#pragma unroll
    for (int d = 0; d < 4; d++) {
      const float4 a0 = *reinterpret_cast<const float4*>(&qs[d][qg * 16]);
      const float4 a1 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 4]);
      const float4 a2 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 8]);
      const float4 a3 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 12]);
      const float4 k0 = *reinterpret_cast<const float4*>(&ks[d][kg * 4]);
      const float4 k1 = *reinterpret_cast<const float4*>(&ks[d][64 + kg * 4]);
      const float qv[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                            a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
      const float kk[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll
      for (int a = 0; a < 4; a++)
#pragma unroll
        for (int h = 0; h < 4; h++)
#pragma unroll
          for (int c = 0; c < 8; c++)
            acc[a][h][c] = fmaf(qv[a * 4 + h], kk[c], acc[a][h][c]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int a = 0; a < 4; a++) {
    const int t = q0 + qg * 4 + a;
    if (t >= count) continue;
    const int lim = min(nb, (first + t + 1) / 4);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const int j = kg * 4 + (c & 3) + 64 * (c >> 2);
      const int b = S + 64 * (j >> 5) + (j & 31);
      if (b < lim) {
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < 4; h++) s += fmaxf(0.f, acc[a][h][c]);
        out[(size_t)t * nb + b] = s * 0.08838834764831845f;
      }
    }
  }
}

int main() {
  struct C { int nb, first, count; } cases[] = {
      {8192, 32768 - 1024, 1024},     // 32K 稳态批
      {20480, 81920 - 1024, 1024},    // 128K 中间批
      {32768, 131072 - 1024, 1024},   // 128K 末批
  };
  const int maxnb = 32768, maxc = 1024;
  float *dq, *dk, *o0, *o1, *o2;
  CK(hipMalloc(&dq, (size_t)maxc * 512 * 4));
  CK(hipMalloc(&dk, (size_t)maxnb * 128 * 4));
  CK(hipMalloc(&o0, (size_t)maxc * maxnb * 4));
  CK(hipMalloc(&o1, (size_t)maxc * maxnb * 4));
  CK(hipMalloc(&o2, (size_t)maxc * maxnb * 4));
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> hq((size_t)maxc * 512), hk((size_t)maxnb * 128);
  for (size_t r = 0; r < maxc * 4; r++)
    for (int d = 0; d < 128; d++) hq[r * 128 + d] = nd(rng) * 0.25f;
  for (size_t r = 0; r < maxnb; r++)
    for (int d = 0; d < 128; d++) hk[r * 128 + d] = nd(rng);
  CK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(dk, hk.data(), hk.size() * 4, hipMemcpyHostToDevice));
  auto t64 = [&](const C& c, float* o) {
    k_index_scores_t64<<<index_t64_grid(c.nb, c.count), 256>>>(dq, dk, o, c.nb,
                                                               c.count, c.first);
  };
  auto s8 = [&](const C& c, float* o) {
    k_index_scores_s8<<<index_t64_grid(c.nb, c.count), 256>>>(dq, dk, o, c.nb,
                                                              c.count, c.first);
  };
  auto s4 = [&](const C& c, float* o) {
    k_index_scores_s4<<<index_t64_grid(c.nb, c.count), 256>>>(dq, dk, o, c.nb,
                                                              c.count, c.first);
  };
  auto time_ms = [&](auto fn, const C& c, float* o, int iters = 20) {
    for (int i = 0; i < 3; i++) fn(c, o);
    CK(hipDeviceSynchronize());
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    CK(hipEventRecord(a));
    for (int i = 0; i < iters; i++) fn(c, o);
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms;
    CK(hipEventElapsedTime(&ms, a, b));
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return ms / iters;
  };
  int fails = 0;
  for (auto& c : cases) {
    const double gf = 2.0 * c.count * 4 * c.nb * 128 / 1e9;
    CK(hipMemset(o0, 0xff, (size_t)c.count * c.nb * 4));
    CK(hipMemset(o1, 0xff, (size_t)c.count * c.nb * 4));
    CK(hipMemset(o2, 0xff, (size_t)c.count * c.nb * 4));
    t64(c, o0); s8(c, o1); s4(c, o2);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    std::vector<unsigned> h0((size_t)c.count * c.nb), h1(h0.size()), h2(h0.size());
    CK(hipMemcpy(h0.data(), o0, h0.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(h1.data(), o1, h1.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(h2.data(), o2, h2.size() * 4, hipMemcpyDeviceToHost));
    size_t d1 = 0, d2 = 0;
    for (size_t i = 0; i < h0.size(); i++) {
      d1 += h0[i] != h1[i];
      d2 += h0[i] != h2[i];
    }
    float m0 = time_ms(t64, c, o0), m1 = time_ms(s8, c, o1), m2 = time_ms(s4, c, o2);
    printf("nb=%-6d count=%-5d | t64 %6.3f ms (%5.2f TF) | s8 %6.3f ms (%5.2f TF)"
           " ndiff=%zu | s4 %6.3f ms (%5.2f TF) ndiff=%zu\n",
           c.nb, c.count, m0, gf / m0, m1, gf / m1, d1, m2, gf / m2, d2);
    if (d1 || d2) fails++;
  }
  printf(fails ? "FAIL\n" : "PASS (bitwise)\n");
  return fails ? 1 : 0;
}
