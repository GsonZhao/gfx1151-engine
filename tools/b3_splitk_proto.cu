// b3_splitk_proto.cu — B3: k1 shape (N=2560, K=6144) split-K microbench.
// Baseline = engine k1 config k_gemm_wmma<64,128,2,2,2,4,64,128> gm=1 (K=6144,
// 96 k-steps). Split variants physically halve/quarter X and W along K into
// contiguous tensors (setup, untimed) and run the same kernel per slice into
// separate f32 Y buffers — the core-loop speedup is measured honestly before
// any epilogue engineering (atomicAdd or second reduce pass) is worth doing.
// Verdict bar: slices' total must reach ~35 TFLOPS-equivalent after allowing
// ~1 ms for the combine pass; otherwise B3 stops here (task spec).
// Kernel is the sed-extracted production one (see gemm_wmma_driver.cu header
// for the regeneration command). Build:
//   hipcc -O3 --offload-arch=gfx1151 -o build/b3_splitk_proto tools/b3_splitk_proto.cu
// Run:   build/b3_splitk_proto [reps=20]   last line: B3 SPLITK: DONE
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                \
  do {                                                                       \
    hipError_t e_ = (x);                                                     \
    if (e_ != hipSuccess) {                                                  \
      fprintf(stderr, "hip error %d (%s) at %s:%d\n", (int)e_,               \
              hipGetErrorString(e_), __FILE__, __LINE__);                    \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

using qw_shortx16 = __attribute__((ext_vector_type(16))) short;
using qw_floatx8 = __attribute__((ext_vector_type(8))) float;
__device__ __forceinline__ qw_shortx16 qw_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  qw_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}

#include "gemm_wmma_kernel.inc"

static uint32_t xs_state;
static uint32_t xs_next() {
  uint32_t x = xs_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  return x;
}
static uint16_t f2bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static void fill_bf16(uint16_t* p, size_t n, uint32_t seed) {
  xs_state = seed;
  for (size_t i = 0; i < n; ++i) {
    const float f = (float)((xs_next() >> 8) & 0xFFFF) / 32768.f - 1.f;
    p[i] = f2bf16(f);
  }
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 20;
  const int N = 2560, K = 6144;
  const int Ps[] = {16384, 8192, 4096};
  const int PMAX = 16384;
  std::vector<uint16_t> hX((size_t)PMAX * K), hW((size_t)N * K);
  fill_bf16(hX.data(), hX.size(), 1);
  fill_bf16(hW.data(), hW.size(), 2);

  uint16_t *X, *W, *Xs[4], *Ws[4];
  float *Y0, *Y1, *Y2, *Y3, *Yref;
  CK(hipMalloc(&X, hX.size() * 2));
  CK(hipMalloc(&W, hW.size() * 2));
  CK(hipMemcpy(X, hX.data(), hX.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(W, hW.data(), hW.size() * 2, hipMemcpyHostToDevice));
  const size_t ybytes = (size_t)PMAX * N * 4;
  CK(hipMalloc(&Y0, ybytes));
  CK(hipMalloc(&Y1, ybytes));
  CK(hipMalloc(&Y2, ybytes));
  CK(hipMalloc(&Y3, ybytes));
  CK(hipMalloc(&Yref, ybytes));

  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  auto med = [&](auto&& f) {
    std::vector<float> ts(reps);
    f();
    CK(hipDeviceSynchronize());
    for (int r = 0; r < reps; ++r) {
      CK(hipEventRecord(e0));
      f();
      CK(hipEventRecord(e1));
      CK(hipEventSynchronize(e1));
      CK(hipEventElapsedTime(&ts[r], e0, e1));
    }
    std::sort(ts.begin(), ts.end());
    return ts[reps / 2];
  };

  for (int P : Ps) {
    const unsigned grid = (unsigned)(((P + 63) / 64) * (N / 128));
    auto base = [&] {
      k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128>
          <<<grid, 128>>>(X, W, Yref, P, K, N, 1);
    };
    const float tb = med(base);
    const double fl = 2.0 * P * (double)N * K;
    printf("P=%5d  base k1(64x128 k64 gm1) %.3f ms  %.1f TF\n", P, tb,
           fl / tb / 1e9);

    // gm variants on the full-K config (cheap extra data point)
    for (int gm : {2, 4, 8}) {
      auto f = [&] {
        k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128>
            <<<grid, 128>>>(X, W, Y0, P, K, N, gm);
      };
      const float t = med(f);
      printf("   full-K gm=%d            %.3f ms  %.1f TF  x%.2f\n", gm, t,
             fl / t / 1e9, tb / t);
    }

    for (int nsplit : {2, 4}) {
      const int ks = K / nsplit;  // 3072 / 1536, both % 64 == 0
      // physical contiguous slices (untimed setup)
      for (int s = 0; s < nsplit; s++) {
        CK(hipMalloc(&Xs[s], (size_t)PMAX * ks * 2));
        CK(hipMalloc(&Ws[s], (size_t)N * ks * 2));
      }
      std::vector<uint16_t> hx((size_t)PMAX * ks), hw((size_t)N * ks);
      for (int s = 0; s < nsplit; s++) {
        for (int r = 0; r < PMAX; r++)
          memcpy(hx.data() + (size_t)r * ks,
                 hX.data() + (size_t)r * K + (size_t)s * ks, ks * 2);
        for (int r = 0; r < N; r++)
          memcpy(hw.data() + (size_t)r * ks,
                 hW.data() + (size_t)r * K + (size_t)s * ks, ks * 2);
        CK(hipMemcpy(Xs[s], hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(Ws[s], hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
      }
      float* Ys[4] = {Y0, Y1, Y2, Y3};
      auto split = [&] {
        for (int s = 0; s < nsplit; s++)
          k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128>
              <<<grid, 128>>>(Xs[s], Ws[s], Ys[s], P, ks, N, 1);
      };
      const float t = med(split);
      printf("   split-%d core (no combine) %.3f ms  %.1f TF-equiv  x%.2f\n",
             nsplit, t, fl / t / 1e9, tb / t);
      // strided variant: no repack, column slices via ldA/ldB (B3 engine plan)
      auto strided = [&] {
        for (int s = 0; s < nsplit; s++)
          k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128>
              <<<grid, 128>>>(X + (size_t)s * ks, W + (size_t)s * ks, Ys[s],
                              P, ks, N, 1, nullptr, nullptr, nullptr, nullptr,
                              (unsigned)K, (unsigned)K);
      };
      const float ts = med(strided);
      printf("   split-%d strided (no repack) %.3f ms  %.1f TF-equiv  x%.2f\n",
             nsplit, ts, fl / ts / 1e9, tb / ts);
      if (nsplit == 2) {
        // mixed: which side needs the contiguous repack?
        auto mixA = [&] {  // contiguous X slices, strided W
          for (int s = 0; s < nsplit; s++)
            k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128>
                <<<grid, 128>>>(Xs[s], W + (size_t)s * ks, Ys[s], P, ks, N, 1,
                                nullptr, nullptr, nullptr, nullptr, 0,
                                (unsigned)K);
        };
        printf("   split-2 contX+stridedW       %.3f ms  %.1f TF-equiv  x%.2f\n",
               med(mixA), fl / med(mixA) / 1e9, tb / med(mixA));
        auto mixB = [&] {  // strided X, contiguous W slices
          for (int s = 0; s < nsplit; s++)
            k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128>
                <<<grid, 128>>>(X + (size_t)s * ks, Ws[s], Ys[s], P, ks, N, 1,
                                nullptr, nullptr, nullptr, nullptr, (unsigned)K,
                                0);
        };
        printf("   split-2 stridedX+contW       %.3f ms  %.1f TF-equiv  x%.2f\n",
               med(mixB), fl / med(mixB) / 1e9, tb / med(mixB));
      }
      // accuracy: sum slices vs full-K reference (f32 accum order differs)
      split();
      base();
      CK(hipDeviceSynchronize());
      std::vector<float> h0((size_t)P * N), hr((size_t)P * N),
          h1((size_t)P * N), h2((size_t)P * N), h3((size_t)P * N);
      CK(hipMemcpy(hr.data(), Yref, (size_t)P * N * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(h0.data(), Y0, (size_t)P * N * 4, hipMemcpyDeviceToHost));
      if (nsplit > 1)
        CK(hipMemcpy(h1.data(), Y1, (size_t)P * N * 4, hipMemcpyDeviceToHost));
      if (nsplit > 2) {
        CK(hipMemcpy(h2.data(), Y2, (size_t)P * N * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h3.data(), Y3, (size_t)P * N * 4, hipMemcpyDeviceToHost));
      }
      double num = 0, den = 0;
      for (size_t i = 0; i < (size_t)P * N; i++) {
        const double s = h0[i] + (nsplit > 1 ? h1[i] : 0) +
                         (nsplit > 2 ? h2[i] + h3[i] : 0);
        num += (s - hr[i]) * (s - hr[i]);
        den += hr[i] * hr[i];
      }
      printf("   split-%d vs full-K rel-L2 %.3e\n", nsplit,
             std::sqrt(num / std::max(den, 1e-30)));
      for (int s = 0; s < nsplit; s++) {
        CK(hipFree(Xs[s]));
        CK(hipFree(Ws[s]));
      }
    }
  }
  printf("B3 SPLITK: DONE\n");
  return 0;
}
