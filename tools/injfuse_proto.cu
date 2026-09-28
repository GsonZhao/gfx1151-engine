// injfuse_proto.cu — HC inject fusion (k_gr_scatter_norm_inj_bf16) vs the
// production k_gr_scatter_norm_b_hc_bf16 + N=4 inject GEMM (d=2560, 4 branches).
//  1. R and Rhat bit-identical to the unfused kernel for the same inject
//     scalar, both scalar sources (pfull = GEMM w4, pin = previous partials),
//     P tails and T variants.
//  2. w4 = inj_psum(partials) vs an fp64 reference of Winj . Rhat(bf16):
//     max error relative to sum|terms|, next to a plain fp32 sequential-K dot
//     (what any GEMM order is in the same class as).
//  3. Timing at P=8192 (old scatter_norm alone; the Lt inject GEMM it replaces
//     is ~0.97 ms/call at P=8192 in the rocprofv3 trace).
// Kernels #included from tools/gr_injfuse_kernel.inc (see tools/injfuse_verify.sh):
//   sed -n '/\[gr-injfuse-begin\]/,/\[gr-injfuse-end\]/p' src/gpu/parts/22_kernels_prefill.inc > tools/gr_injfuse_kernel.inc
// Build: hipcc -O3 --offload-arch=gfx1151 -o build/injfuse_proto tools/injfuse_proto.cu
// Run:   build/injfuse_proto [reps=20]     last line: INJFUSE PROTO: PASS/FAIL
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
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

// same as src/gpu/parts/22_kernels_prefill.inc
__device__ inline uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
__device__ inline float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

#include "gr_injfuse_kernel.inc"

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static float urand() {  // [-1, 1)
  rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
  return (float)((rs >> 40) * (1.0 / (1ull << 23))) - 1.f;
}
static uint16_t h_f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static float h_bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

template <int T>
static void launch_new(const float* pf, const float* pi, uint16_t* R, const float* y,
                       const float* w, uint16_t* Rh, const uint16_t* wi, float* po,
                       int d, int P, float eps) {
  k_gr_scatter_norm_inj_bf16<T><<<4 * ((P + T - 1) / T), 256>>>(pf, pi, R, y, w, Rh,
                                                                    wi, po, d, P, eps);
}
typedef void (*NewFn)(const float*, const float*, uint16_t*, const float*, const float*,
                      uint16_t*, const uint16_t*, float*, int, int, float);

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 20;
  const int d = 2560, NB = 4, K = NB * d, PM = 8192;
  const float eps = 1e-6f;
  bool ok = true;
  std::vector<uint16_t> hR((size_t)PM * K), hWi((size_t)4 * K);
  std::vector<float> hy((size_t)PM * d), hw((size_t)NB * d), hw4((size_t)PM * 4),
      hpin((size_t)PM * 16);
  for (auto& v : hR) v = h_f2bf(3.f * urand());
  for (int t : {9, 4001})  // extreme-scale rows
    for (int c = 0; c < K; ++c) hR[(size_t)t * K + c] = h_f2bf(h_bf2f(hR[(size_t)t * K + c]) * (t == 9 ? 1e-3f : 200.f));
  for (auto& v : hy) v = 1.5f * urand();
  for (auto& v : hw) v = 0.3f * urand();
  for (auto& v : hWi) v = h_f2bf(0.05f * urand());
  for (auto& v : hw4) v = 6.f * urand();
  for (auto& v : hpin) v = 3.f * urand();
  uint16_t *R0, *R1, *R2, *Rh1, *Rh2, *Wi;
  float *y, *w, *w4, *pin, *po, *w4s;
  CK(hipMalloc(&R0, hR.size() * 2));
  CK(hipMalloc(&R1, hR.size() * 2));
  CK(hipMalloc(&R2, hR.size() * 2));
  CK(hipMalloc(&Rh1, hR.size() * 2));
  CK(hipMalloc(&Rh2, hR.size() * 2));
  CK(hipMalloc(&Wi, hWi.size() * 2));
  CK(hipMalloc(&y, hy.size() * 4));
  CK(hipMalloc(&w, hw.size() * 4));
  CK(hipMalloc(&w4, hw4.size() * 4));
  CK(hipMalloc(&w4s, hw4.size() * 4));
  CK(hipMalloc(&pin, hpin.size() * 4));
  CK(hipMalloc(&po, hpin.size() * 4));
  CK(hipMemcpy(R0, hR.data(), hR.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(Wi, hWi.data(), hWi.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(y, hy.data(), hy.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(w, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(w4, hw4.data(), hw4.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(pin, hpin.data(), hpin.size() * 4, hipMemcpyHostToDevice));
  CK(hipGetLastError());

  struct V { int T; NewFn f; } vars[] = {{4, launch_new<4>}, {8, launch_new<8>}, {16, launch_new<16>}};
  std::vector<uint16_t> a((size_t)PM * K), b((size_t)PM * K), ra((size_t)PM * K), rb((size_t)PM * K);
  std::vector<float> hpo((size_t)PM * 16);
  const int Ps[] = {1, 3, 1024, 2051, 8192};
  double worst_rel = 0, worst_rel_seq = 0;
  for (int P : Ps) {
    const size_t n = (size_t)P * K;
    for (int src = 0; src < 2; ++src) {  // 0: pfull=w4, 1: pin partials
      const float* w4ref = w4;
      if (src) {
        k_inj_psum<<<(P * 4 + 255) / 256, 256>>>(pin, w4s, P);
        w4ref = w4s;
      }
      CK(hipMemcpy(R1, R0, n * 2, hipMemcpyDeviceToDevice));
      CK(hipMemset(Rh1, 0xCD, n * 2));
      k_gr_scatter_norm_b_hc_bf16<<<4 * P, 256>>>(w4ref, R1, y, w, Rh1, d, NB, P, eps);
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(ra.data(), R1, n * 2, hipMemcpyDeviceToHost));
      CK(hipMemcpy(a.data(), Rh1, n * 2, hipMemcpyDeviceToHost));
      for (auto& v : vars) {
        CK(hipMemcpy(R2, R0, n * 2, hipMemcpyDeviceToDevice));
        CK(hipMemset(Rh2, 0xAB, n * 2));
        CK(hipMemset(po, 0xFF, (size_t)P * 16 * 4));
        v.f(src ? nullptr : w4, src ? pin : nullptr, R2, y, w, Rh2, Wi, po, d, P, eps);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(rb.data(), R2, n * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), Rh2, n * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hpo.data(), po, (size_t)P * 16 * 4, hipMemcpyDeviceToHost));
        size_t mr = 0, mh = 0;
        for (size_t i = 0; i < n; ++i) { mr += ra[i] != rb[i]; mh += a[i] != b[i]; }
        // w4 accuracy on the kernel's own Rhat output (bf16 as stored).
        double rel = 0, rel_seq = 0;
        bool nanbad = false;
        for (int t = 0; t < P; ++t)
          for (int k = 0; k < 4; ++k) {
            double ref = 0, mag = 0;
            float seq = 0.f;
            for (int c = 0; c < K; ++c) {
              double term = (double)h_bf2f(b[(size_t)t * K + c]) * h_bf2f(hWi[(size_t)k * K + c]);
              ref += term;
              mag += fabs(term);
              seq += h_bf2f(b[(size_t)t * K + c]) * h_bf2f(hWi[(size_t)k * K + c]);
            }
            float got = hpo[(size_t)t * 16 + 0 * 4 + k];
            got += hpo[(size_t)t * 16 + 1 * 4 + k];
            got += hpo[(size_t)t * 16 + 2 * 4 + k];
            got += hpo[(size_t)t * 16 + 3 * 4 + k];
            if (!std::isfinite(got)) nanbad = true;
            if (mag > 0) {
              rel = std::max(rel, fabs(got - ref) / mag);
              rel_seq = std::max(rel_seq, fabs(seq - ref) / mag);
            }
          }
        worst_rel = std::max(worst_rel, rel);
        worst_rel_seq = std::max(worst_rel_seq, rel_seq);
        const bool pass = !mr && !mh && !nanbad && rel < 1e-5;
        printf("P=%-5d src=%s T=%-2d  R mism %zu  Rhat mism %zu  w4 err/sum|terms| %.2e (fp32 seq %.2e)%s\n",
               P, src ? "pin  " : "pfull", v.T, mr, mh, rel, rel_seq, pass ? "" : "  <-- FAIL");
        if (!pass) ok = false;
      }
    }
  }
  // timing at P=8192
  {
    const int P = 8192;
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    auto med = [&](auto fn) {
      std::vector<float> ts;
      for (int r = 0; r < reps; ++r) {
        CK(hipMemcpy(R2, R0, (size_t)P * K * 2, hipMemcpyDeviceToDevice));
        float m;
        CK(hipEventRecord(e0)); fn(); CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&m, e0, e1));
        ts.push_back(m);
      }
      std::sort(ts.begin(), ts.end());
      return ts[reps / 2];
    };
    float t_old = med([&] { k_gr_scatter_norm_b_hc_bf16<<<4 * P, 256>>>(w4, R2, y, w, Rh2, d, NB, P, eps); });
    printf("P=8192 timing: old scatter_norm %.3f ms (+ Lt inject GEMM ~0.97 ms in trace)\n", t_old);
    for (auto& v : vars) {
      float t = med([&] { v.f(nullptr, pin, R2, y, w, Rh2, Wi, po, d, P, eps); });
      printf("               fused T=%-2d %.3f ms  -> saves ~%.3f ms/call vs old+GEMM\n", v.T, t,
             t_old + 0.97f - t);
    }
  }
  printf("worst w4 err/sum|terms|: fused %.2e, fp32 sequential %.2e\n", worst_rel, worst_rel_seq);
  printf("INJFUSE PROTO: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
