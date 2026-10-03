// moe_lut_test.cu — standalone check + bench of the LUT-decoded 4-bit routed
// expert WMMA kernels (src/gpu/parts/27_kernels_moe_lut.inc).
//
//   moe_lut_test --hgn W4B.hgn [--layer L]         real hgn q4cp experts
//   moe_lut_test --v2 V2.hgn [--layer L]           real hgn v2 i4r experts
//                                                  (up = raw gate|up, down fed
//                                                  random f16 hid, both isolated)
//   moe_lut_test --synth iq4nl|iq4xs [--E N]       synthetic GGUF IQ4 blocks
//                                                  (gate/up that type, down IQ4_NL)
//   moe_lut_test --hgn-dense W4B.hgn --tensor NAME  A3 prototype: one dense
//                                                  q4cp weight as a single
//                                                  "expert" of the LUT down
//                                                  GEMM vs dequant + k_gemm_wmma
//   common: [--P N] [--iters N] [--check-experts N] [--seed S] [--tol X]
//           [--q4perm 1]  hgn only: register-perm codebook decode (A1);
//                         bit-compared against the s_cbp path, both timed
//
// Routing: random top-10 (distinct experts per token, mildly skewed), like
// tools/moe_gguf_test.cu. Correctness: outputs are pre-filled with NaN and
// must be fully written; for --check-experts experts ALL their slots are
// compared with a CPU double reference built from the format's own
// dequantizer (hgn.h q4cp_row / gguf.h dequant_row):
//   up:   hid = up * silu(gate) from the F16-rounded x
//   down: pairs from the GPU's F16 hid (isolates the down kernel)
// Metric: per-slot relative L2 error; FAIL if max > --tol (default 1e-2).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../src/gguf.h"
#include "../src/hgn.h"

struct MoeTile { int expert, first, count; };
#include "../src/gpu/parts/26_kernels_moe_gguf.inc"
#include "../src/gpu/parts/27_kernels_moe_lut.inc"

static uint16_t bf16_bits_host(float f) {  // RNE bit pattern, same as f2bf
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static float bf16_val_host(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// Verbatim copy of k_i4r_mid (28_kernels_hgn_v2.inc) as the --mid reference;
// 28 itself is not included here (it needs helpers from 22_kernels_prefill.inc).
__global__ __launch_bounds__(512) void k_mid_ref(const __half* __restrict__ guv, uint64_t gs,
                                                 void* out, uint64_t os,
                                                 const float* __restrict__ svh_gu,
                                                 const float* __restrict__ suh_dn, int mid) {
  __shared__ float s[2048];
  const int slot = blockIdx.x;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  {
    const __half* z = guv + (size_t)slot * gs + w * 128;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = (float)z[j * 32 + lane];
    hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int i = w * 128 + j * 32 + lane;
      s[i] = v[j] * hv2::kRs128 * svh_gu[i];
    }
  }
  __syncthreads();
  if (w >= (mid >> 7)) return;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = w * 128 + j * 32 + lane;
    const float g = s[i], u = s[mid + i];
    v[j] = g / (1.f + expf(-g)) * u * suh_dn[i];
  }
  hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = w * 128 + j * 32 + lane;
    ((__half*)out)[(size_t)slot * os + i] = __float2half(v[j] * hv2::kRs128);
  }
}

#define CK(x)                                                                    \
  do {                                                                           \
    hipError_t e_ = (x);                                                         \
    if (e_ != hipSuccess) {                                                      \
      fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                   \
    }                                                                            \
  } while (0)

// A3 prototype (--hgn-dense): k_dequant_q4cp_bf16 + k_gemm_wmma live in 22,
// which needs g_cfg (05) for an unrelated rope kernel plus the paged-QSA
// addressing helpers below (verbatim from 10_ple_io.inc).
__device__ __forceinline__ int qsa_source_pos(int slot, int visible,
                                              const int* blocks) {
  if (!blocks || visible < 2052) return slot;
  return slot < 2048 ? 4 * blocks[slot / 4] + slot % 4
                     : (visible / 4) * 4 + slot - 2048;
}
constexpr int KV_PAGE_SHIFT = 8;
constexpr int KV_PAGE = 1 << KV_PAGE_SHIFT;
constexpr int KV_PAGE_BLOCKS = KV_PAGE / 4;
constexpr int KV_PAGE_BLOCK_SHIFT = KV_PAGE_SHIFT - 2;
__device__ __forceinline__ int kv_row(const int* __restrict__ ptab, int t) {
  return ptab ? (ptab[t >> KV_PAGE_SHIFT] << KV_PAGE_SHIFT) | (t & (KV_PAGE - 1))
              : t;
}
__device__ __forceinline__ int ik_blk(const int* __restrict__ ptab, int b) {
  return ptab ? (ptab[b >> KV_PAGE_BLOCK_SHIFT] << KV_PAGE_BLOCK_SHIFT) |
                    (b & (KV_PAGE_BLOCKS - 1))
              : b;
}
__device__ __forceinline__ float4 kv_ld4(const float* __restrict__ p) {
  return *(const float4*)p;
}
__device__ __forceinline__ float4 kv_ld4(const uint16_t* __restrict__ p) {
  uint2 pk = *(const uint2*)p;  // 4 bf16, little-endian pairs
  uint32_t x0 = pk.x << 16, x1 = pk.x & 0xffff0000u;
  uint32_t x2 = pk.y << 16, x3 = pk.y & 0xffff0000u;
  float4 v;
  memcpy(&v.x, &x0, 4);
  memcpy(&v.y, &x1, 4);
  memcpy(&v.z, &x2, 4);
  memcpy(&v.w, &x3, 4);
  return v;
}
#include "../src/gpu/parts/05_config.inc"
#include "../src/gpu/parts/22_kernels_prefill.inc"

// ---------------------------------------------------------------------------
// Prototype i4r-only k_moe_lut variant for HANDOFF-V2-PREFILL direction 4.
// OPT bits: 1 = arithmetic nibble decode (no s_cbp LDS table),
//           2 = i4r scale kept in registers across the 2 stages it covers,
//           4 = BK=4 (one 128-element scale block per stage, half the syncs),
//           8 = two row fragments per wave (staged rows doubled, B fragments
//               read from LDS once per 2 rows).
// i4r specifics folded in unconditionally: one scale per row per stage (BK=2
// stages always lie inside one 128-element block), stored as a single
// s_scale[row] entry instead of BK copies.
template <unsigned OPT, bool kPair>
__launch_bounds__(256) __global__
    void k_moe_lut_opt(moelut::LutW p, const __half* __restrict__ x, const int* __restrict__ tokidx,
                       const MoeTile* __restrict__ tiles, const int* __restrict__ ntiles,
                       float* __restrict__ out, __half* __restrict__ out_half, int m, int k) {
  using namespace moegg;
  using namespace moelut;
  constexpr bool kArith = (OPT & 1) != 0;
  constexpr bool kSCache = (OPT & 2) != 0;
  constexpr int BK = (OPT & 4) ? 4 : 2;
  constexpr bool kDual = (OPT & 8) != 0;
  constexpr int kFetchKb = BK / 2;            // kb columns per thread per row
  constexpr int kR2 = kDual ? 2 : 1;          // row fragments per thread / wave
  constexpr int BM = 128 * kR2;               // staged rows
  constexpr int BN = 64;
  constexpr int kTokTiles = BN / 16;
  constexpr int kCodeBytes = BM * BK * 16;
  constexpr int kScaleBytes = BM * 4;
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  constexpr int kStageBytes = kCodeBytes + kScaleBytes + kActBytes;
  constexpr int kPairLds = kPair ? 8 * kR2 * 16 * 18 * 4 : 0;
  constexpr int kLdsBytes0 = kStageBytes > kPairLds ? kStageBytes : kPairLds;
  constexpr int kLdsBytes = kLdsBytes0 > 8 * 1024 ? kLdsBytes0 : 8 * 1024;
  __shared__ __attribute__((aligned(16))) uint8_t lds[kLdsBytes];
  __shared__ uint32_t s_cbp[kArith ? 1 : 256];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_scale = reinterpret_cast<uint32_t*>(lds + kCodeBytes);
  auto* s_act = reinterpret_cast<uint4*>(lds + kCodeBytes + kScaleBytes);

  if ((int)blockIdx.y >= *ntiles) return;
  const int tid = threadIdx.x;
  if constexpr (!kArith) {
    const float lo = p.cb[tid & 15], hi = p.cb[tid >> 4];
    s_cbp[tid] = __builtin_bit_cast(uint32_t, __floats2half2_rn(lo, hi));
  }

  const MoeTile tile = tiles[blockIdx.y];
  const int expert = tile.expert;
  const int slot0 = tile.first;
  const int nrows = tile.count;  // <= BN
  const int live_tok_tiles = (nrows + 15) / 16;
  const int num_kb = k / 32;
  const size_t rb = (size_t)k / 2;

  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = (int)blockIdx.x * kRows;

  // Weight fetch: thread = (row tid/2 [+ 128 if kDual], K blocks kb0 +
  // (tid&1)*kFetchKb .. +kFetchKb-1).
  const int f_r0 = tid >> 1;
  const int f_k0 = (tid & 1) * kFetchKb;
  const uint8_t* f_ptr[kR2];
  const uint8_t* f_sp[kR2];
  bool f_live[kR2];
#pragma unroll
  for (int r2 = 0; r2 < kR2; ++r2) {
    const int fr = f_r0 + r2 * 128;
    const bool upper = kPair && fr >= kRows;
    const int r = r_block + (kPair ? fr % kRows : fr);
    f_live[r2] = r < m;
    const size_t row = (size_t)expert * p.e_rows + (upper ? p.up_off : 0) + (f_live[r2] ? r : (m - 1));
    f_ptr[r2] = (upper ? p.w_up : p.w) + row * rb;
    f_sp[r2] = (upper ? p.sc_up : p.sc) + row * (size_t)p.sstride;
  }
  uint4 f_codes[kR2][kFetchKb];
  uint32_t f_s[kR2];
  constexpr int kActFetch = BK;
  constexpr int kActChunks = BN * BK * 4;
  uint4 a_data[kActFetch];
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + i * 256;
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    int src = -1;
    if (chunk < kActChunks && t < nrows) src = tokidx ? tokidx[slot0 + t] : slot0 + t;
    a_src[i] = src >= 0 ? x + (size_t)src * k + sub * 8 : nullptr;
    a_slot[i] = chunk < kActChunks ? sub * kActStride + t : -1;
  }

  const auto swizzle = [](int row, int c) { return row * BK + (c ^ ((row >> 2) & 1)); };

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int r2 = 0; r2 < kR2; ++r2) {
#pragma unroll
      for (int i = 0; i < kFetchKb; ++i)
        f_codes[r2][i] = *reinterpret_cast<const uint4*>(f_ptr[r2] + (size_t)(kb0 + f_k0 + i) * 16);
      if (!kSCache || (kb0 & 3) == 0)
        f_s[r2] = *reinterpret_cast<const uint16_t*>(f_sp[r2] + (size_t)(kb0 >> 2) * 2);
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      a_data[i] = a_src[i] ? *reinterpret_cast<const uint4*>(a_src[i] + kb0 * 32)
                           : make_uint4(0u, 0u, 0u, 0u);
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int r2 = 0; r2 < kR2; ++r2) {
      const int row = f_r0 + r2 * 128;
#pragma unroll
      for (int i = 0; i < kFetchKb; ++i) s_codes[swizzle(row, f_k0 + i)] = f_codes[r2][i];
      s_scale[row] = f_live[r2] ? (f_s[r2] | (f_s[r2] << 16)) : 0U;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      if (a_slot[i] >= 0) s_act[a_slot[i]] = a_data[i];
  };

  v8f acc[kR2][kTokTiles];
#pragma unroll
  for (int r2 = 0; r2 < kR2; ++r2)
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) acc[r2][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

  const auto compute_stage = [&]() {
    const int row0 = wave_id * 16 + sub_lane;
    __half2 scale2[kR2];
#pragma unroll
    for (int r2 = 0; r2 < kR2; ++r2)
      scale2[r2] = __builtin_bit_cast(__half2, s_scale[row0 + r2 * 128]);
    uint4 raw[kR2][BK];
#pragma unroll
    for (int r2 = 0; r2 < kR2; ++r2)
#pragma unroll
      for (int c = 0; c < BK; ++c) raw[r2][c] = s_codes[swizzle(row0 + r2 * 128, c)];
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      __half2 h[kR2][16];
#pragma unroll
      for (int r2 = 0; r2 < kR2; ++r2) {
        const uint32_t words[4] = {raw[r2][kb].x, raw[r2][kb].y, raw[r2][kb].z, raw[r2][kb].w};
        if constexpr (kArith) {
          // f16 bits 0x6400+nib = 1024+nib (exact); minus 1032 -> nib-8
          // (exact, Sterbenz); * scale rounds like the LDS-table hmul2.
          const __half2 bias = __builtin_bit_cast(__half2, 0x64086408U);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint32_t w = words[i];
            const uint32_t lo = w & 0x0F0F0F0FU;
            const uint32_t hi = (w >> 4) & 0x0F0F0F0FU;
            const uint32_t p0 = __builtin_amdgcn_perm(hi, lo, 0x05040100U);  // [l0,l1,h0,h1]
            const uint32_t p1 = __builtin_amdgcn_perm(hi, lo, 0x07060302U);  // [l2,l3,h2,h3]
            const uint32_t q[4] = {p0, p0 >> 8, p1, p1 >> 8};
#pragma unroll
            for (int j2 = 0; j2 < 4; ++j2)
              h[r2][4 * i + j2] =
                  __hmul2(__hsub2(__builtin_bit_cast(__half2, (q[j2] & 0x00FF00FFU) | 0x64006400U), bias),
                          scale2[r2]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const uint32_t b = (words[i >> 2] >> (8 * (i & 3))) & 0xFFU;
            h[r2][i] = __hmul2(__builtin_bit_cast(__half2, s_cbp[b]), scale2[r2]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        asm volatile("" ::: "memory");  // keep one token tile's fragments live
        if (j >= live_tok_tiles) continue;
        const uint4* frag = s_act + (kb * 4) * kActStride + j * 16 + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) b[q] = frag[q * kActStride];
        v16h b_lo, b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int r2 = 0; r2 < kR2; ++r2) {
          v16h a_lo, a_hi;
          __builtin_memcpy(&a_lo, &h[r2][0], 32);
          __builtin_memcpy(&a_hi, &h[r2][8], 32);
          acc[r2][j] = wmma(a_lo, b_lo, acc[r2][j]);
          acc[r2][j] = wmma(a_hi, b_hi, acc[r2][j]);
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) fetch_stage(kb0 + BK);
    compute_stage();
    __syncthreads();
  }

  if constexpr (kPair) {
    // Gate rows live in staged rows [0, kRows), up rows in [kRows, BM); wave w
    // holds staged rows w*16..+16 and (with kR2 == 2) 128 + w*16..+16, so a
    // wave always pairs its own gate/up rows. Two LDS planes per wave.
    constexpr unsigned stride = 18;
    constexpr unsigned plane = 16 * stride;
    // Up rows sit kR2==1: in waves 4-7 (+4 planes); kR2==2: in each wave's
    // second plane (+1 plane).
    constexpr unsigned upoff = kR2 == 1 ? 4 * plane : plane;
    float* base = reinterpret_cast<float*>(lds);
    float* scratch = base + wave_id * kR2 * plane;
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      if (j >= live_tok_tiles) break;  // uniform across the block
#pragma unroll
      for (int r2 = 0; r2 < kR2; ++r2)
#pragma unroll
        for (int l = 0; l < 8; ++l)
          scratch[r2 * plane + sub_lane * stride + 2 * l + half_id] = acc[r2][j][l];
      __syncthreads();
#pragma unroll
      for (int unit = 0; unit < kRows / 32; ++unit) {
        const int flat = (unit * 256 + tid) * 2;
        const int t = j * 16 + flat / kRows;
        const int r = flat % kRows;
        if (t < nrows && r_block + r < m) {
          const int idx = (r / 16) * kR2 * plane + (flat / kRows) * stride + r % 16;
          const size_t o = (size_t)(slot0 + t) * 2 * m + r_block + r;
          *reinterpret_cast<__half2*>(out_half + o) = __floats2half2_rn(base[idx], base[idx + 1]);
          *reinterpret_cast<__half2*>(out_half + o + m) =
              __floats2half2_rn(base[idx + upoff], base[idx + 1 + upoff]);
        }
      }
      __syncthreads();
    }
    return;
  }

  // Down: transpose each 16x16 tile through LDS; one 64 B row segment per slot.
  float* tile_scratch = reinterpret_cast<float*>(lds) + wave_id * 256;
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    if (j >= live_tok_tiles) break;
#pragma unroll
    for (int r2 = 0; r2 < kR2; ++r2) {
      const int r0 = r_block + r2 * 128 + wave_id * 16;
#pragma unroll
      for (int l = 0; l < 8; ++l) tile_scratch[sub_lane * 16 + 2 * l + half_id] = acc[r2][j][l];
      __builtin_amdgcn_wave_barrier();
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = s * 32 + lane_id;
        const int t = j * 16 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < nrows && r < m) out[(size_t)(slot0 + t) * m + r] = tile_scratch[flat];
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

static bool moe_lut_up_opt(int opt, const moelut::LutW& p, const __half* x, const int* tokidx,
                           const MoeTile* tiles, const int* ntiles, int max_tiles, __half* guv, int m,
                           int k, hipStream_t st) {
  dim3 grid(m / ((opt & 8) ? 128 : 64), max_tiles);
  switch (opt) {
#define UP_CASE(O) \
  case O: k_moe_lut_opt<O, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, guv, m, k); break
    UP_CASE(0); UP_CASE(1); UP_CASE(2); UP_CASE(3); UP_CASE(4); UP_CASE(5); UP_CASE(6); UP_CASE(7);
    UP_CASE(8); UP_CASE(9); UP_CASE(10); UP_CASE(11); UP_CASE(12); UP_CASE(13); UP_CASE(14); UP_CASE(15);
#undef UP_CASE
    default: return false;
  }
  return true;
}

static bool moe_lut_down_opt(int opt, const moelut::LutW& p, const __half* hid, const MoeTile* tiles,
                             const int* ntiles, int max_tiles, float* pairs, int m, int k,
                             hipStream_t st) {
  dim3 grid(m / ((opt & 8) ? 256 : 128), max_tiles);
  switch (opt) {
#define DN_CASE(O) \
  case O: k_moe_lut_opt<O, false><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k); break
    DN_CASE(0); DN_CASE(1); DN_CASE(2); DN_CASE(3); DN_CASE(4); DN_CASE(5); DN_CASE(6); DN_CASE(7);
    DN_CASE(8); DN_CASE(9); DN_CASE(10); DN_CASE(11); DN_CASE(12); DN_CASE(13); DN_CASE(14); DN_CASE(15);
#undef DN_CASE
    default: return false;
  }
  return true;
}


static float h2f(__half h) { return __half2float(h); }
static uint8_t* upload(const void* src, size_t n);

// --hgn-dense (A3 prototype): one dense q4cp weight [N][K] run as a
// single-"expert" LUT down GEMM (decode fused into the A loader, f32 out)
// versus the engine's dense path: k_dequant_q4cp_bf16 -> k_gemm_wmma. x is
// rounded to f16 for the LUT side and bf16 for the WMMA side, matching the
// engine's own converters; the f32->f16/bf16 x conversion itself is shared
// by both in-engine and excluded here.
static int dense_mode(const std::string& path, const std::string& tensor, int P,
                      int iters, unsigned seed, double tol) {
  if (tensor.empty()) {
    fprintf(stderr, "--hgn-dense requires --tensor\n");
    return 2;
  }
  hgn::Checkpoint ck(path.c_str());
  const hgn::Tensor& t = ck.at(tensor.c_str());
  const auto q = hgn::Checkpoint::q4cp_parse(t);
  const int N = (int)q.rows, K = (int)q.cols;
  printf("dense %s: q4cp [%d][%d] scale_stride=%zu, P=%d\n", tensor.c_str(), N, K,
         (size_t)q.scale_stride, P);
  const uint8_t* dw = upload(t.data, t.data_size);
  moelut::LutW pdn = lut_q4cp_view(dw, q.rows, q.cols, q.scale_stride, N, 0);
  if (!moelut::lut_ok(moelut::kQ4CP, pdn, false, K)) {
    fprintf(stderr, "dense: lut_ok(kQ4CP, down, K=%d) failed\n", K);
    return 2;
  }
  const int gw_kind = (N == 2560 && K == 6144) ? 1
                    : (K == 2560 && (N == 6144 || N == 10240 || N == 12288)) ? 2
                    : (N == 320 && K == 10240) ? 3 : 0;
  if (!gw_kind) {
    fprintf(stderr, "dense: no k_gemm_wmma config for %dx%d\n", N, K);
    return 2;
  }

  std::mt19937 drng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<__half> xh((size_t)P * K);
  std::vector<uint16_t> xb((size_t)P * K);
  for (size_t i = 0; i < xh.size(); i++) {
    const float v = nd(drng);
    xh[i] = __float2half(v);
    xb[i] = bf16_bits_host(v);
  }
  std::vector<MoeTile> tiles;
  for (int r = 0; r < P; r += 64) tiles.push_back({0, r, std::min(64, P - r)});
  const int ntiles = (int)tiles.size(), max_tiles = (P + 63) / 64 + 1;

  __half* d_x;
  uint16_t *d_xb, *d_wbf;
  float *d_out, *d_y;
  int* d_nt;
  MoeTile* d_tiles;
  CK(hipMalloc(&d_x, xh.size() * 2));
  CK(hipMalloc(&d_xb, xb.size() * 2));
  CK(hipMalloc(&d_wbf, (size_t)N * K * 2));
  CK(hipMalloc(&d_out, (size_t)P * N * 4));
  CK(hipMalloc(&d_y, (size_t)P * N * 4));
  CK(hipMalloc(&d_nt, 4));
  CK(hipMalloc(&d_tiles, (size_t)max_tiles * sizeof(MoeTile)));
  CK(hipMemcpy(d_x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_xb, xb.data(), xb.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_nt, &ntiles, 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_tiles, tiles.data(), ntiles * sizeof(MoeTile), hipMemcpyHostToDevice));
  CK(hipMemset(d_out, 0xFF, (size_t)P * N * 4));
  CK(hipMemset(d_y, 0xFF, (size_t)P * N * 4));

  auto run_lut = [&] {
    if (!moe_lut_down(moelut::kQ4CP, pdn, d_x, d_tiles, d_nt, max_tiles, d_out, N, K,
                      0)) {
      fprintf(stderr, "dense: moe_lut_down unsupported\n");
      exit(2);
    }
  };
  auto run_deq = [&] {
    const uint64_t tot = (uint64_t)N * K / 32;
    k_dequant_q4cp_bf16<<<(unsigned)((tot + 255) / 256), 256>>>(
        dw + 64, dw + 64 + (size_t)N * K / 2, (const float*)dw, d_wbf, N, K,
        q.scale_stride);
  };
  auto run_gemm = [&] {
    if (gw_kind == 1)
      k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128>
          <<<(unsigned)(((P + 63) / 64) * 20), 128>>>(d_xb, d_wbf, d_y, P, K, N, 1);
    else if (gw_kind == 2)
      k_gemm_wmma<128, 256, 2, 4, 4, 4, 32, 256>
          <<<(unsigned)(((P + 127) / 128) * ((N + 255) / 256)), 256>>>(d_xb, d_wbf,
                                                                       d_y, P, K, N, 4);
    else
      k_gemm_wmma<64, 160, 2, 2, 2, 5, 64, 128>
          <<<(unsigned)(((P + 63) / 64) * 2), 128>>>(d_xb, d_wbf, d_y, P, K, N, 1);
  };
  run_lut();
  run_deq();
  run_gemm();
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());

  // correctness: 4 tokens x 64 output rows, per-(t,r) relative L2 vs the
  // double reference from q4cp_row and the path's own x rounding
  std::vector<float> out_h((size_t)P * N), y_h((size_t)P * N);
  CK(hipMemcpy(out_h.data(), d_out, out_h.size() * 4, hipMemcpyDeviceToHost));
  CK(hipMemcpy(y_h.data(), d_y, y_h.size() * 4, hipMemcpyDeviceToHost));
  bool ok = true;
  size_t nan_lut = 0, nan_g = 0;
  for (float v : out_h) nan_lut += !std::isfinite(v);
  for (float v : y_h) nan_g += !std::isfinite(v);
  printf("coverage: non-finite lut %zu / %zu, gemm %zu / %zu\n", nan_lut, out_h.size(),
         nan_g, y_h.size());
  if (nan_lut || nan_g) ok = false;
  double lut_max = 0, g_max = 0;
  const int toks[4] = {0, P / 3, 2 * P / 3, P - 1};
  std::vector<float> wr(K);
  for (int ti = 0; ti < 4; ti++) {
    const int tt = toks[ti];
    for (int ri = 0; ri < 64; ri++) {
      const int r = (int)((ri * 7919 + 13) % N);
      hgn::Checkpoint::q4cp_row(q, r, wr.data());
      // error relative to the root-sum-square of the per-column products:
      // |ref| itself is ~N(0, rss) and cancels near zero, which would blow up
      // a plain |g-ref|/|ref| on unlucky draws
      double num_l = 0, den_l = 0, num_g = 0, den_g = 0;
      double ref_l = 0, ref_g = 0;
      for (int c = 0; c < K; c++) {
        const double pl = wr[c] * h2f(xh[(size_t)tt * K + c]);
        const double pg = wr[c] * bf16_val_host(xb[(size_t)tt * K + c]);
        ref_l += pl;
        ref_g += pg;
        den_l += pl * pl;
        den_g += pg * pg;
      }
      const double gl = out_h[(size_t)tt * N + r], gg = y_h[(size_t)tt * N + r];
      num_l = (gl - ref_l) * (gl - ref_l);
      num_g = (gg - ref_g) * (gg - ref_g);
      lut_max = std::max(lut_max, std::sqrt(num_l / std::max(den_l, 1e-30)));
      g_max = std::max(g_max, std::sqrt(num_g / std::max(den_g, 1e-30)));
    }
  }
  printf("check: 4 tokens x 64 rows | lut rel-L2 max %.3e | gemm rel-L2 max %.3e (tol %.1e)\n",
         lut_max, g_max, tol);
  if (!(lut_max <= tol) || !(g_max <= tol)) ok = false;

  hipEvent_t e0, e1, e2, e3;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  CK(hipEventCreate(&e2));
  CK(hipEventCreate(&e3));
  float t_lut = 0, t_deq = 0, t_gemm = 0;
  for (int it = 0; it < iters; it++) {
    CK(hipEventRecord(e0, 0));
    run_lut();
    CK(hipEventRecord(e1, 0));
    run_deq();
    CK(hipEventRecord(e2, 0));
    run_gemm();
    CK(hipEventRecord(e3, 0));
    CK(hipEventSynchronize(e3));
    float a, b, c;
    CK(hipEventElapsedTime(&a, e0, e1));
    CK(hipEventElapsedTime(&b, e1, e2));
    CK(hipEventElapsedTime(&c, e2, e3));
    if (it > 0 || iters == 1) { t_lut += a; t_deq += b; t_gemm += c; }
  }
  const int nt = iters > 1 ? iters - 1 : 1;
  t_lut /= nt;
  t_deq /= nt;
  t_gemm /= nt;
  const double fl = 2.0 * P * (double)N * K;
  const double wbytes = 64.0 + (double)N * K / 2 + (double)N * q.scale_stride;
  printf("time lut  : %.3f ms (%.1f TFLOPS, weight %.1f GB/s effective)\n", t_lut,
         fl / t_lut / 1e9, wbytes / t_lut / 1e6);
  printf("time deq+ : dequant %.3f ms + gemm %.3f ms (%.1f TFLOPS) = %.3f ms\n", t_deq,
         t_gemm, fl / t_gemm / 1e9, t_deq + t_gemm);
  printf("a3 ratio  : (deq+gemm) / lut = %.2fx  (%s at P=%d)\n",
         (t_deq + t_gemm) / t_lut, (t_deq + t_gemm) / t_lut > 1.0 ? "lut wins" : "deq+gemm wins",
         P);
  printf("%s\n", ok ? "RESULT PASS" : "RESULT FAIL");
  return ok ? 0 : 1;
}

static uint8_t* upload(const void* src, size_t n) {
  uint8_t* d;
  CK(hipMalloc(&d, n));
  CK(hipMemcpy(d, src, n, hipMemcpyHostToDevice));
  return d;
}

static uint16_t f2h_bits(float f) { return __builtin_bit_cast(uint16_t, __float2half(f)); }

// hgn v2 i4r (dtype 23): codes plane [rows][cols/2] (byte b = element 2b |
// 2b+1 << 4), then scales plane [rows][cols/128] fp16, rows unpadded.
static void i4r_row(const uint8_t* data, uint64_t rows, uint64_t cols, uint64_t r, float* out) {
  const uint8_t* codes = data + r * (cols / 2);
  const uint8_t* sc = data + rows * (cols / 2) + r * (cols / 128 * 2);
  for (uint64_t c = 0; c < cols; c++) {
    const uint8_t b = codes[c >> 1];
    const int nib = (c & 1) ? b >> 4 : b & 15;
    uint16_t sb;
    memcpy(&sb, sc + (c >> 7) * 2, 2);
    out[c] = (float)(nib - 8) * hgn::fp16_to_f32(sb);
  }
}

// Synthetic IQ4 expert tensor [E][rows][cols] with plausible scales.
static std::vector<uint8_t> synth_iq4(int type, size_t E, size_t rows, size_t cols, std::mt19937& rng) {
  const size_t rb = type == moelut::kIQ4NL ? cols / 32 * 18 : cols / 256 * 136;
  std::vector<uint8_t> v(E * rows * rb);
  std::uniform_int_distribution<int> byte(0, 255), ls(0, 63);
  std::uniform_real_distribution<float> dd(0.2f, 1.0f);
  for (size_t r = 0; r < E * rows; r++) {
    uint8_t* p = v.data() + r * rb;
    if (type == moelut::kIQ4NL) {
      for (size_t b = 0; b < cols / 32; b++, p += 18) {
        uint16_t d = f2h_bits(dd(rng) * 2e-4f * (byte(rng) & 1 ? 1.f : -1.f));
        memcpy(p, &d, 2);
        for (int j = 0; j < 16; j++) p[2 + j] = (uint8_t)byte(rng);
      }
    } else {
      for (size_t b = 0; b < cols / 256; b++, p += 136) {
        uint16_t d = f2h_bits(dd(rng) * 1e-5f);
        memcpy(p, &d, 2);
        uint16_t sh = 0;
        uint8_t sl[4] = {0, 0, 0, 0};
        for (int ib = 0; ib < 8; ib++) {
          const int s = ls(rng);
          sl[ib / 2] |= (uint8_t)((s & 0xF) << (4 * (ib % 2)));
          sh |= (uint16_t)(((s >> 4) & 3) << (2 * ib));
        }
        memcpy(p + 2, &sh, 2);
        memcpy(p + 4, sl, 4);
        for (int j = 0; j < 128; j++) p[8 + j] = (uint8_t)byte(rng);
      }
    }
  }
  return v;
}

int main(int argc, char** argv) {
  std::string hgn_path, v2_path, synth, dense_path, tensor;
  int layer = 0, P = 2048, iters = 10, check_e = 8, topk = 10, E_synth = 128;
  int opt = -1;     // >= 0 (v2 only): prototype kernel with this OPT bitmask
  int fuse_mid = 0; // v2 only: fused up+mid (moe_lut_up_mid) writing hid
  int q4perm = 0;   // hgn only: register-perm codebook decode (A1), A/B bit-compared
  unsigned seed = 1;
  double tol = 1e-2;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--hgn") hgn_path = argv[i + 1];
    else if (a == "--v2") v2_path = argv[i + 1];
    else if (a == "--synth") synth = argv[i + 1];
    else if (a == "--hgn-dense") dense_path = argv[i + 1];
    else if (a == "--tensor") tensor = argv[i + 1];
    else if (a == "--opt") opt = atoi(argv[i + 1]);
    else if (a == "--mid") fuse_mid = atoi(argv[i + 1]);
    else if (a == "--q4perm") q4perm = atoi(argv[i + 1]);
    else if (a == "--layer") layer = atoi(argv[i + 1]);
    else if (a == "--P") P = atoi(argv[i + 1]);
    else if (a == "--E") E_synth = atoi(argv[i + 1]);
    else if (a == "--iters") iters = atoi(argv[i + 1]);
    else if (a == "--check-experts") check_e = atoi(argv[i + 1]);
    else if (a == "--seed") seed = (unsigned)atoi(argv[i + 1]);
    else if (a == "--tol") tol = atof(argv[i + 1]);
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  const int nmodes = !hgn_path.empty() + !v2_path.empty() + !synth.empty() + !dense_path.empty();
  if (nmodes != 1) {
    fprintf(stderr,
            "usage: %s (--hgn W4B.hgn | --v2 V2.hgn [--layer L] | --synth iq4nl|iq4xs [--E N] | --hgn-dense W4B.hgn --tensor NAME) [--P N] ...\n",
            argv[0]);
    return 2;
  }
  if (!dense_path.empty()) return dense_mode(dense_path, tensor, P, iters, seed, tol);
  if (opt >= 0 && v2_path.empty()) {
    fprintf(stderr, "--opt requires --v2\n");
    return 2;
  }
  if (fuse_mid && (v2_path.empty() || opt >= 0)) {
    fprintf(stderr, "--mid requires --v2 and no --opt\n");
    return 2;
  }
  if (q4perm && (hgn_path.empty() || opt >= 0)) {
    fprintf(stderr, "--q4perm requires --hgn and no --opt\n");
    return 2;
  }
  std::mt19937 rng(seed);

  int D, MID, E, t_gu, t_dn_type;
  moelut::LutW pgu{}, pdn{};
  float *d_svh = nullptr, *d_suh = nullptr;  // v2 sidecars (f32 device copies)
  std::vector<float> svh_h, suh_h;           // and their host copies
  // CPU dequant of (expert, row) into out[cols]
  std::function<void(int, int, float*)> deq_gate, deq_up, deq_down;
  std::unique_ptr<hgn::Checkpoint> ck;
  std::vector<uint8_t> hg, hu, hd;  // synthetic host copies
  if (!hgn_path.empty()) {
    ck.reset(new hgn::Checkpoint(hgn_path.c_str()));
    char nm[128];
    snprintf(nm, sizeof nm, "layers.%d.mlp.experts.gate_up_proj.weight", layer);
    const hgn::Tensor& tgu = ck->at(nm);
    snprintf(nm, sizeof nm, "layers.%d.mlp.experts.down_proj.weight", layer);
    const hgn::Tensor& tdn = ck->at(nm);
    const auto qgu = hgn::Checkpoint::q4cp_parse(tgu);
    const auto qdn = hgn::Checkpoint::q4cp_parse(tdn);
    E = (int)tgu.dims[0];
    D = (int)qgu.cols;
    MID = (int)(qgu.rows / E / 2);
    if (qdn.cols != (uint64_t)MID || qdn.rows != (uint64_t)E * D) {
      fprintf(stderr, "unexpected shapes\n");
      return 2;
    }
    t_gu = t_dn_type = moelut::kQ4CP;
    const uint8_t* dgu = upload(tgu.data, tgu.data_size);
    const uint8_t* ddn = upload(tdn.data, tdn.data_size);
    pgu = lut_q4cp_view(dgu, qgu.rows, qgu.cols, qgu.scale_stride, 2 * MID, MID);
    pdn = lut_q4cp_view(ddn, qdn.rows, qdn.cols, qdn.scale_stride, D, 0);
    deq_gate = [=](int e, int r, float* o) { hgn::Checkpoint::q4cp_row(qgu, (uint64_t)e * 2 * MID + r, o); };
    deq_up = [=](int e, int r, float* o) { hgn::Checkpoint::q4cp_row(qgu, (uint64_t)e * 2 * MID + MID + r, o); };
    deq_down = [=](int e, int r, float* o) { hgn::Checkpoint::q4cp_row(qdn, (uint64_t)e * D + r, o); };
    printf("hgn layer %d: q4cp gate_up [%d][%d][%d], down [%d][%d][%d], P=%d top%d\n", layer, E, 2 * MID, D,
           E, D, MID, P, topk);
  } else if (!v2_path.empty()) {
    ck.reset(new hgn::Checkpoint(v2_path.c_str()));
    char nm[128];
    snprintf(nm, sizeof nm, "layers.%d.mlp.experts.gate_up_proj.weight", layer);
    const hgn::Tensor& tgu = ck->at(nm);
    snprintf(nm, sizeof nm, "layers.%d.mlp.experts.down_proj.weight", layer);
    const hgn::Tensor& tdn = ck->at(nm);
    if (tgu.dtype != 23 || tdn.dtype != 23) {
      fprintf(stderr, "v2: expected dtype 23 experts, got %u / %u\n", tgu.dtype, tdn.dtype);
      return 2;
    }
    E = (int)tgu.dims[0];
    D = (int)tgu.dims[tgu.ndims - 1];
    const uint64_t rows_gu = tgu.numel() / (uint64_t)D;
    MID = (int)(rows_gu / (uint64_t)E / 2);
    const uint64_t rows_dn = tdn.numel() / (uint64_t)MID;
    if (tdn.dims[tdn.ndims - 1] != (uint64_t)MID || rows_dn != (uint64_t)E * D || D % 128 || MID % 128) {
      fprintf(stderr, "v2: unexpected shapes\n");
      return 2;
    }
    t_gu = t_dn_type = moelut::kI4R;
    const float cb[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
    float* d_cb;
    CK(hipMalloc(&d_cb, sizeof cb));
    CK(hipMemcpy(d_cb, cb, sizeof cb, hipMemcpyHostToDevice));
    const uint8_t* dgu = upload(tgu.data, tgu.data_size);
    const uint8_t* ddn = upload(tdn.data, tdn.data_size);
    pgu = lut_i4r_view(dgu, rows_gu, (uint64_t)D, d_cb, 2 * MID, MID);
    pdn = lut_i4r_view(ddn, rows_dn, (uint64_t)MID, d_cb, D, 0);
    deq_gate = [=](int e, int r, float* o) { i4r_row(tgu.data, rows_gu, D, (uint64_t)e * 2 * MID + r, o); };
    deq_up = [=](int e, int r, float* o) { i4r_row(tgu.data, rows_gu, D, (uint64_t)e * 2 * MID + MID + r, o); };
    deq_down = [=](int e, int r, float* o) { i4r_row(tdn.data, rows_dn, MID, (uint64_t)e * D + r, o); };
    if (fuse_mid) {
      auto load_sidecar = [&](const char* suffix, size_t want, std::vector<float>& h, float** d) {
        snprintf(nm, sizeof nm, "layers.%d.mlp.experts.%s", layer, suffix);
        const hgn::Tensor& t = ck->at(nm);
        if (t.dtype != 2 || t.numel() != want) {
          fprintf(stderr, "v2: bad sidecar %s (dtype %u, numel %llu, want %zu)\n", nm, t.dtype,
                  (unsigned long long)t.numel(), want);
          exit(2);
        }
        h.resize(want);
        const uint16_t* src = (const uint16_t*)t.data;
        for (size_t i = 0; i < want; i++) h[i] = hgn::fp16_to_f32(src[i]);
        CK(hipMalloc(d, want * 4));
        CK(hipMemcpy(*d, h.data(), want * 4, hipMemcpyHostToDevice));
      };
      load_sidecar("gate_up_proj.svh", (size_t)2 * MID, svh_h, &d_svh);
      load_sidecar("down_proj.suh", (size_t)MID, suh_h, &d_suh);
    }
    printf("v2 layer %d: i4r gate_up [%d][%d][%d], down [%d][%d][%d], P=%d top%d\n", layer, E, 2 * MID, D, E,
           D, MID, P, topk);
  } else {
    D = 2560;
    MID = 640;
    E = E_synth;
    if (synth == "iq4nl") t_gu = moelut::kIQ4NL;
    else if (synth == "iq4xs") t_gu = moelut::kIQ4XS;
    else { fprintf(stderr, "--synth iq4nl|iq4xs\n"); return 2; }
    t_dn_type = moelut::kIQ4NL;  // 640 is not a multiple of 256
    hg = synth_iq4(t_gu, E, MID, D, rng);
    hu = synth_iq4(t_gu, E, MID, D, rng);
    hd = synth_iq4(t_dn_type, E, D, MID, rng);
    pgu.w = upload(hg.data(), hg.size());
    pgu.w_up = upload(hu.data(), hu.size());
    pgu.e_rows = MID;
    pdn.w = upload(hd.data(), hd.size());
    pdn.e_rows = D;
    const size_t rbg = t_gu == moelut::kIQ4NL ? D / 32 * 18 : D / 256 * 136, rbd = MID / 32 * 18;
    const uint32_t tg = (uint32_t)t_gu;
    deq_gate = [&, rbg, tg](int e, int r, float* o) { gguf::dequant_row(tg, hg.data() + ((size_t)e * MID + r) * rbg, o, D); };
    deq_up = [&, rbg, tg](int e, int r, float* o) { gguf::dequant_row(tg, hu.data() + ((size_t)e * MID + r) * rbg, o, D); };
    deq_down = [&, rbd](int e, int r, float* o) { gguf::dequant_row(gguf::IQ4_NL, hd.data() + ((size_t)e * D + r) * rbd, o, MID); };
    printf("synthetic: gate/up %s [%d][%d][%d], down IQ4_NL [%d][%d][%d], P=%d top%d\n",
           t_gu == moelut::kIQ4NL ? "IQ4_NL" : "IQ4_XS", E, MID, D, E, D, MID, P, topk);
  }

  // routing (mild skew: expert popularity ~ 1/(1+e/64))
  std::vector<double> pop(E);
  for (int e = 0; e < E; e++) pop[e] = 1.0 / (1.0 + ((e * 2654435761u) % E) / 64.0);
  std::discrete_distribution<int> pick(pop.begin(), pop.end());
  std::vector<int> ids((size_t)P * topk);
  for (int t = 0; t < P; t++) {
    for (int s = 0; s < topk; s++) {
      int e;
      bool dup;
      do {
        e = pick(rng);
        dup = false;
        for (int q = 0; q < s; q++) dup |= ids[(size_t)t * topk + q] == e;
      } while (dup);
      ids[(size_t)t * topk + s] = e;
    }
  }
  const int npairs = P * topk;
  std::vector<int> eoff(E + 1, 0), tokidx(npairs);
  for (int p = 0; p < npairs; p++) eoff[ids[p] + 1]++;
  for (int e = 0; e < E; e++) eoff[e + 1] += eoff[e];
  {
    std::vector<int> cur(eoff.begin(), eoff.end() - 1);
    for (int t = 0; t < P; t++)
      for (int s = 0; s < topk; s++) tokidx[cur[ids[(size_t)t * topk + s]]++] = t;
  }
  std::vector<MoeTile> tiles;
  for (int e = 0; e < E; e++)
    for (int r = eoff[e]; r < eoff[e + 1]; r += 64) tiles.push_back({e, r, std::min(64, eoff[e + 1] - r)});
  const int ntiles = (int)tiles.size();
  const int max_tiles = (npairs + 63) / 64 + E;  // engine's grid bound

  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<__half> xh((size_t)P * D);
  for (auto& v : xh) v = __float2half(nd(rng));

  const bool v2 = !v2_path.empty();
  // v2: down is checked/timed in isolation on random f16 hid (like the engine,
  // where hid comes from k_i4r_mid, not from this kernel).
  std::vector<__half> hid_in;
  if (v2) {
    hid_in.resize((size_t)npairs * MID);
    for (auto& v : hid_in) v = __float2half(nd(rng) * 0.05f);
  }

  __half *d_x, *d_hid, *d_guv = nullptr, *d_hidmid = nullptr;
  float* d_pairs;
  int *d_tok, *d_nt;
  MoeTile* d_tiles;
  CK(hipMalloc(&d_x, xh.size() * 2));
  CK(hipMalloc(&d_hid, (size_t)npairs * MID * 2));
  if (v2) CK(hipMalloc(&d_guv, (size_t)npairs * 2 * MID * 2));
  if (fuse_mid) CK(hipMalloc(&d_hidmid, (size_t)npairs * MID * 2));
  CK(hipMalloc(&d_pairs, (size_t)npairs * D * 4));
  CK(hipMalloc(&d_tok, npairs * 4));
  CK(hipMalloc(&d_nt, 4));
  CK(hipMalloc(&d_tiles, (size_t)max_tiles * sizeof(MoeTile)));
  CK(hipMemcpy(d_x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_tok, tokidx.data(), npairs * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_nt, &ntiles, 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_tiles, tiles.data(), ntiles * sizeof(MoeTile), hipMemcpyHostToDevice));
  if (v2) {
    CK(hipMemcpy(d_hid, hid_in.data(), hid_in.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemset(fuse_mid ? d_hidmid : d_guv, 0xFF,
                 (size_t)npairs * (fuse_mid ? MID : 2 * MID) * 2));  // NaN
  } else {
    CK(hipMemset(d_hid, 0xFF, (size_t)npairs * MID * 2));  // NaN
  }
  CK(hipMemset(d_pairs, 0xFF, (size_t)npairs * D * 4));  // NaN

  auto run_up = [&](bool perm) {
    if (v2) {
      const bool launched =
          fuse_mid
              ? moe_lut_up_mid(pgu, d_x, d_tok, d_tiles, d_nt, max_tiles, d_hidmid, d_svh, d_suh, MID,
                               D, 0)
              : opt >= 0
                  ? moe_lut_up_opt(opt, pgu, d_x, d_tok, d_tiles, d_nt, max_tiles, d_guv, MID, D, 0)
                  : moe_lut_up_raw(pgu, d_x, d_tok, d_tiles, d_nt, max_tiles, d_guv, MID, D, 0);
      if (!launched) {
        fprintf(stderr, "up: unsupported\n");
        exit(2);
      }
      return;
    }
    const bool launched =
        perm ? moe_lut_up_perm(pgu, d_x, d_tok, d_tiles, d_nt, max_tiles, d_hid, MID, D, 0)
             : moe_lut_up(t_gu, pgu, d_x, d_tok, d_tiles, d_nt, max_tiles, d_hid, MID, D, 0);
    if (!launched) {
      fprintf(stderr, "moe_lut_up: unsupported\n");
      exit(2);
    }
  };
  auto run_dn = [&](bool perm) {
    const bool launched =
        opt >= 0
            ? moe_lut_down_opt(opt, pdn, d_hid, d_tiles, d_nt, max_tiles, d_pairs, D, MID, 0)
            : perm ? moe_lut_down_perm(pdn, d_hid, d_tiles, d_nt, max_tiles, d_pairs, D, MID, 0)
                   : moe_lut_down(t_dn_type, pdn, d_hid, d_tiles, d_nt, max_tiles, d_pairs, D, MID, 0);
    if (!launched) {
      fprintf(stderr, "down: unsupported\n");
      exit(2);
    }
  };
  run_up(q4perm != 0);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
  run_dn(q4perm != 0);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());

  // --q4perm: device A/B of the register-perm decode against the s_cbp table
  // path. Independent chains (baseline down is fed the baseline hid); both hid
  // and pairs must be bit-identical.
  if (q4perm) {
    __half* d_hid0;
    float* d_pairs0;
    CK(hipMalloc(&d_hid0, (size_t)npairs * MID * 2));
    CK(hipMalloc(&d_pairs0, (size_t)npairs * D * 4));
    CK(hipMemset(d_hid0, 0xFF, (size_t)npairs * MID * 2));
    CK(hipMemset(d_pairs0, 0xFF, (size_t)npairs * D * 4));
    if (!moe_lut_up(t_gu, pgu, d_x, d_tok, d_tiles, d_nt, max_tiles, d_hid0, MID, D, 0) ||
        !moe_lut_down(t_dn_type, pdn, d_hid0, d_tiles, d_nt, max_tiles, d_pairs0, D, MID, 0)) {
      fprintf(stderr, "q4perm baseline: unsupported\n");
      return 2;
    }
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    std::vector<__half> href((size_t)npairs * MID), hf((size_t)npairs * MID);
    std::vector<float> pref((size_t)npairs * D), pf((size_t)npairs * D);
    CK(hipMemcpy(href.data(), d_hid0, href.size() * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hf.data(), d_hid, hf.size() * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(pref.data(), d_pairs0, pref.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(pf.data(), d_pairs, pf.size() * 4, hipMemcpyDeviceToHost));
    CK(hipFree(d_hid0));
    CK(hipFree(d_pairs0));
    size_t bad = 0, first = SIZE_MAX;
    for (size_t i = 0; i < href.size(); i++)
      if (__builtin_bit_cast(uint16_t, href[i]) != __builtin_bit_cast(uint16_t, hf[i])) {
        if (first == SIZE_MAX) first = i;
        bad++;
      }
    printf("perm vs s_cbp hid: %zu / %zu mismatch", bad, href.size());
    if (bad) {
      const size_t slot = first / MID, r = first % MID;
      printf(" (first at slot %zu row %zu: perm %g ref %g)", slot, r, h2f(hf[first]),
             h2f(href[first]));
    }
    printf("\n");
    bool ok_bits = bad == 0;
    bad = 0;
    first = SIZE_MAX;
    for (size_t i = 0; i < pref.size(); i++)
      if (__builtin_bit_cast(uint32_t, pref[i]) != __builtin_bit_cast(uint32_t, pf[i])) {
        if (first == SIZE_MAX) first = i;
        bad++;
      }
    printf("perm vs s_cbp pairs: %zu / %zu mismatch", bad, pref.size());
    if (bad) {
      const size_t slot = first / D, r = first % D;
      printf(" (first at slot %zu row %zu: perm %g ref %g)", slot, r, pf[first], pref[first]);
    }
    printf("\n");
    if (bad) ok_bits = false;
    if (!ok_bits) {
      printf("RESULT FAIL\n");
      return 1;
    }
  }

  // --mid: device A/B of the fused kernel against up_raw + k_mid_ref (the
  // exact pair it replaces). Must be bit-identical.
  if (fuse_mid) {
    __half* d_hidref;
    CK(hipMalloc(&d_hidref, (size_t)npairs * MID * 2));
    CK(hipMemset(d_hidref, 0xFF, (size_t)npairs * MID * 2));
    if (!moe_lut_up_raw(pgu, d_x, d_tok, d_tiles, d_nt, max_tiles, d_guv, MID, D, 0)) {
      fprintf(stderr, "up_raw: unsupported\n");
      return 2;
    }
    k_mid_ref<<<npairs, 2 * MID / 128 * 32, 0, 0>>>(d_guv, 2 * MID, d_hidref, MID, d_svh, d_suh,
                                                    MID);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    std::vector<__half> href((size_t)npairs * MID), hf((size_t)npairs * MID);
    CK(hipMemcpy(href.data(), d_hidref, href.size() * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hf.data(), d_hidmid, hf.size() * 2, hipMemcpyDeviceToHost));
    size_t bad = 0, first = SIZE_MAX;
    for (size_t i = 0; i < href.size(); i++)
      if (__builtin_bit_cast(uint16_t, href[i]) != __builtin_bit_cast(uint16_t, hf[i])) {
        if (first == SIZE_MAX) first = i;
        bad++;
      }
    printf("fused vs up_raw+k_i4r_mid: %zu / %zu mismatch", bad, href.size());
    if (bad) {
      const size_t slot = first / MID, r = first % MID;
      printf(" (first at slot %zu row %zu: fused %g ref %g)", slot, r, h2f(hf[first]),
             h2f(href[first]));
    }
    printf("\n");
    CK(hipFree(d_hidref));
    if (bad) {
      printf("RESULT FAIL\n");
      return 1;
    }
  }

  std::vector<__half> hid(v2 ? 0 : (size_t)npairs * MID);
  std::vector<__half> guv(v2 && !fuse_mid ? (size_t)npairs * 2 * MID : 0);
  std::vector<__half> hidm(fuse_mid ? (size_t)npairs * MID : 0);
  std::vector<float> pairs((size_t)npairs * D);
  if (fuse_mid) CK(hipMemcpy(hidm.data(), d_hidmid, hidm.size() * 2, hipMemcpyDeviceToHost));
  else if (v2) CK(hipMemcpy(guv.data(), d_guv, guv.size() * 2, hipMemcpyDeviceToHost));
  else CK(hipMemcpy(hid.data(), d_hid, hid.size() * 2, hipMemcpyDeviceToHost));
  CK(hipMemcpy(pairs.data(), d_pairs, pairs.size() * 4, hipMemcpyDeviceToHost));
  bool ok = true;
  size_t nan_h = 0, nan_p = 0;
  for (auto v : guv) nan_h += !std::isfinite(h2f(v));
  for (auto v : hidm) nan_h += !std::isfinite(h2f(v));
  for (auto v : hid) nan_h += !std::isfinite(h2f(v));
  for (auto v : pairs) nan_p += !std::isfinite(v);
  printf("coverage: non-finite %s %zu / %zu, pairs %zu / %zu\n", fuse_mid ? "hidm" : v2 ? "guv" : "hid",
         nan_h, fuse_mid ? hidm.size() : v2 ? guv.size() : hid.size(), nan_p, pairs.size());
  if (nan_h || nan_p) ok = false;

  std::vector<int> ce;
  for (int i = 0; i < E && (int)ce.size() < check_e; i++) {
    int e = (int)(((uint64_t)i * 7919 + 13) % E);
    if (eoff[e + 1] > eoff[e] && std::find(ce.begin(), ce.end(), e) == ce.end()) ce.push_back(e);
  }
  double up_max = 0, up_sum = 0, dn_max = 0, dn_sum = 0, ref_rms = 0;
  int nchk = 0;
  std::vector<float> wg((size_t)MID * D), wu((size_t)MID * D), wd((size_t)D * MID);
  for (int e : ce) {
    for (int r = 0; r < MID; r++) {
      deq_gate(e, r, &wg[(size_t)r * D]);
      deq_up(e, r, &wu[(size_t)r * D]);
    }
    for (int r = 0; r < D; r++) deq_down(e, r, &wd[(size_t)r * MID]);
    for (int sl = eoff[e]; sl < eoff[e + 1]; sl++) {
      const __half* xr = &xh[(size_t)tokidx[sl] * D];
      std::vector<double> xf(D);
      for (int c = 0; c < D; c++) xf[c] = h2f(xr[c]);
      double num = 0, den = 0;
      if (v2 && fuse_mid) {
        // Full reference of the fused up+mid: dots -> f16 (the guv16 rounding)
        // -> per-128 FWHT -> *kRs128*svh -> silu(g)*u*suh_dn -> FWHT -> *kRs128.
        std::vector<double> gg(MID), uu(MID);
        for (int r = 0; r < MID; r++) {
          double g = 0, u = 0;
          const float* pg = &wg[(size_t)r * D];
          const float* pu = &wu[(size_t)r * D];
          for (int c = 0; c < D; c++) { g += pg[c] * xf[c]; u += pu[c] * xf[c]; }
          gg[r] = h2f(__float2half((float)g));
          uu[r] = h2f(__float2half((float)u));
        }
        const auto fwht = [](double* v) {
          for (int s = 1; s < 128; s <<= 1)
            for (int i = 0; i < 128; i += 2 * s)
              for (int j = i; j < i + s; j++) {
                const double a = v[j], b = v[j + s];
                v[j] = a + b;
                v[j + s] = a - b;
              }
        };
        constexpr double kRs = 0.0883883476483184405;
        for (int b = 0; b < MID / 128; b++) {
          double vg[128], vu[128], hv[128];
          for (int i = 0; i < 128; i++) { vg[i] = gg[b * 128 + i]; vu[i] = uu[b * 128 + i]; }
          fwht(vg);
          fwht(vu);
          for (int i = 0; i < 128; i++) {
            const double g2 = vg[i] * kRs * svh_h[b * 128 + i];
            const double u2 = vu[i] * kRs * svh_h[MID + b * 128 + i];
            hv[i] = g2 / (1.0 + std::exp(-g2)) * u2 * suh_h[b * 128 + i];
          }
          fwht(hv);
          for (int i = 0; i < 128; i++) {
            const double ref = hv[i] * kRs;
            const double got = h2f(hidm[(size_t)sl * MID + b * 128 + i]);
            num += (got - ref) * (got - ref);
            den += ref * ref;
          }
        }
        ref_rms += std::sqrt(den / MID);
      } else
      for (int r = 0; r < MID; r++) {
        double g = 0, u = 0;
        const float* pg = &wg[(size_t)r * D];
        const float* pu = &wu[(size_t)r * D];
        for (int c = 0; c < D; c++) { g += pg[c] * xf[c]; u += pu[c] * xf[c]; }
        if (v2) {
          const double got_g = h2f(guv[(size_t)sl * 2 * MID + r]);
          const double got_u = h2f(guv[(size_t)sl * 2 * MID + MID + r]);
          num += (got_g - g) * (got_g - g) + (got_u - u) * (got_u - u);
          den += g * g + u * u;
          ref_rms += std::sqrt((g * g + u * u) / 2);
          continue;
        }
        double ref = u * g / (1.0 + std::exp(-g));
        double got = h2f(hid[(size_t)sl * MID + r]);
        num += (got - ref) * (got - ref);
        den += ref * ref;
        if (r == MID - 1) ref_rms += std::sqrt(den / MID);
      }
      double rel = std::sqrt(num / std::max(den, 1e-30));
      up_max = std::max(up_max, rel);
      up_sum += rel;
      num = den = 0;
      std::vector<double> hf(MID);
      const std::vector<__half>& hsrc = v2 ? hid_in : hid;
      for (int c = 0; c < MID; c++) hf[c] = h2f(hsrc[(size_t)sl * MID + c]);
      for (int r = 0; r < D; r++) {
        double y = 0;
        const float* pd = &wd[(size_t)r * MID];
        for (int c = 0; c < MID; c++) y += pd[c] * hf[c];
        double got = pairs[(size_t)sl * D + r];
        num += (got - y) * (got - y);
        den += y * y;
      }
      rel = std::sqrt(num / std::max(den, 1e-30));
      dn_max = std::max(dn_max, rel);
      dn_sum += rel;
      nchk++;
    }
  }
  printf("check: %zu experts, %d slots | hid rms %.3e | up rel-L2 max %.3e mean %.3e | down rel-L2 max %.3e mean %.3e (tol %.1e)\n",
         ce.size(), nchk, ref_rms / std::max(nchk, 1), up_max, up_sum / std::max(nchk, 1), dn_max,
         dn_sum / std::max(nchk, 1), tol);
  if (!(up_max <= tol) || !(dn_max <= tol) || nchk == 0) ok = false;

  hipEvent_t e0, e1, e2;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  CK(hipEventCreate(&e2));
  auto bench = [&](bool perm, float& t_up, float& t_dn) {
    t_up = 0;
    t_dn = 0;
    for (int it = 0; it < iters; it++) {
      CK(hipEventRecord(e0, 0));
      run_up(perm);
      CK(hipEventRecord(e1, 0));
      run_dn(perm);
      CK(hipEventRecord(e2, 0));
      CK(hipEventSynchronize(e2));
      float a, b;
      CK(hipEventElapsedTime(&a, e0, e1));
      CK(hipEventElapsedTime(&b, e1, e2));
      if (it > 0 || iters == 1) { t_up += a; t_dn += b; }
    }
    int nt = iters > 1 ? iters - 1 : 1;
    t_up /= nt;
    t_dn /= nt;
  };
  double fl_up = 2.0 * npairs * 2 * MID * (double)D, fl_dn = 2.0 * npairs * (double)D * MID;
  if (q4perm) {
    float b_up, b_dn, p_up, p_dn;
    bench(false, b_up, b_dn);
    bench(true, p_up, p_dn);
    printf("time s_cbp: up %.3f ms (%.1f TFLOPS)  down %.3f ms (%.1f TFLOPS)  total %.3f ms/layer = %.4f ms/tok over 48 layers\n",
           b_up, fl_up / b_up / 1e9, b_dn, fl_dn / b_dn / 1e9, b_up + b_dn, (b_up + b_dn) * 48 / P);
    printf("time perm : up %.3f ms (%.1f TFLOPS)  down %.3f ms (%.1f TFLOPS)  total %.3f ms/layer = %.4f ms/tok over 48 layers\n",
           p_up, fl_up / p_up / 1e9, p_dn, fl_dn / p_dn / 1e9, p_up + p_dn, (p_up + p_dn) * 48 / P);
    printf("perm gain : up %+.2f%%  down %+.2f%%  total %+.2f%%\n",
           100.0 * (b_up - p_up) / b_up, 100.0 * (b_dn - p_dn) / b_dn,
           100.0 * (b_up + b_dn - p_up - p_dn) / (b_up + b_dn));
  } else {
    float t_up, t_dn;
    bench(false, t_up, t_dn);
    printf("time: up %.3f ms (%.1f TFLOPS)  down %.3f ms (%.1f TFLOPS)  total %.3f ms/layer = %.4f ms/tok over 48 layers\n",
           t_up, fl_up / t_up / 1e9, t_dn, fl_dn / t_dn / 1e9, t_up + t_dn, (t_up + t_dn) * 48 / P);
  }
  printf("%s\n", ok ? "RESULT PASS" : "RESULT FAIL");
  return ok ? 0 : 1;
}
