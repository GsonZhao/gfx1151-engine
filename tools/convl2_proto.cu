// convl2_proto.cu — GDN conv+silu + q/k L2 norm fusion (k_gdn_conv_l2n_b) vs
// the production pair k_gdn_conv_b -> k_l2norm_qk_b (in place), qkv = 10240
// (16 q heads + 16 k heads of 128, then 6144 v). Checks the fused output is
// bit-identical over all P x 10240 elements (P tails, convst history rows,
// a few huge / tiny rows), and times both.
// The kernels are #included from tools/gdn_convl2_kernel.inc (sed-extracted
// from src/gpu/parts/25_kernels_gdn.inc, see tools/convl2_verify.sh):
//   sed -n '/\[gdn-convl2-begin\]/,/\[gdn-convl2-end\]/p' src/gpu/parts/25_kernels_gdn.inc > tools/gdn_convl2_kernel.inc
// Build: hipcc -O3 --offload-arch=gfx1151 -o build/convl2_proto tools/convl2_proto.cu
// Run:   build/convl2_proto [reps=20]     last line: CONVL2 PROTO: PASS/FAIL
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

#include "gdn_convl2_kernel.inc"

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static float urand() {  // [-1, 1)
  rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
  return (float)((rs >> 40) * (1.0 / (1ull << 23))) - 1.f;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 20;
  const int QKV = 10240, NQK = 16;
  const float qscale = 1.f / sqrtf(128.f);
  const int Ps[] = {1, 3, 1024, 2051, 8192};
  bool ok = true;
  const int PM = 8192;
  float *x, *cw, *cs, *o1, *o2;
  CK(hipMalloc(&x, (size_t)PM * QKV * 4));
  CK(hipMalloc(&o1, (size_t)PM * QKV * 4));
  CK(hipMalloc(&o2, (size_t)PM * QKV * 4));
  CK(hipMalloc(&cw, (size_t)QKV * 4 * 4));
  CK(hipMalloc(&cs, (size_t)3 * QKV * 4));
  std::vector<float> hx((size_t)PM * QKV), hcw((size_t)QKV * 4), hcs((size_t)3 * QKV);
  for (auto& v : hx) v = 2.5f * urand();
  for (int t : {5, 777, 4000})  // a few rows with extreme scale
    for (int c = 0; c < QKV; ++c) hx[(size_t)t * QKV + c] *= (t == 777 ? 1e-4f : 300.f);
  for (auto& v : hcw) v = 0.6f * urand();
  for (auto& v : hcs) v = 2.f * urand();
  CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(cw, hcw.data(), hcw.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(cs, hcs.data(), hcs.size() * 4, hipMemcpyHostToDevice));
  std::vector<uint32_t> a((size_t)PM * QKV), b((size_t)PM * QKV);
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  for (int P : Ps) {
    auto run_old = [&]() {
      int64_t tot = (int64_t)P * QKV / 4;
      k_gdn_conv_b<<<(unsigned)((tot + 255) / 256), 256>>>(x, cw, cs, o1, P, QKV);
      k_l2norm_qk_b<<<dim3(NQK, P), 128>>>(o1, 128, qscale, QKV, NQK);
    };
    auto run_new = [&]() {
      k_gdn_conv_l2n_b<<<dim3((QKV / 128 + 7) / 8, P), 256>>>(x, cw, cs, o2, P, QKV, NQK,
                                                                 qscale);
    };
    CK(hipMemset(o1, 0xCD, (size_t)PM * QKV * 4));
    CK(hipMemset(o2, 0xAB, (size_t)PM * QKV * 4));
    run_old();
    run_new();
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(a.data(), o1, (size_t)P * QKV * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), o2, (size_t)P * QKV * 4, hipMemcpyDeviceToHost));
    size_t mism = 0, first = (size_t)-1;
    for (size_t i = 0; i < (size_t)P * QKV; ++i)
      if (a[i] != b[i]) { if (!mism) first = i; ++mism; }
    float ms_o = 0, ms_n = 0;
    if (P >= 1024) {
      std::vector<float> to, tn;
      for (int r = 0; r < reps; ++r) {
        float m;
        CK(hipEventRecord(e0)); run_old(); CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&m, e0, e1)); to.push_back(m);
        CK(hipEventRecord(e0)); run_new(); CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&m, e0, e1)); tn.push_back(m);
      }
      std::sort(to.begin(), to.end());
      std::sort(tn.begin(), tn.end());
      ms_o = to[reps / 2];
      ms_n = tn[reps / 2];
    }
    printf("P=%-5d mismatch %zu", P, mism);
    if (mism) printf(" (first at t=%zu ch=%zu)", first / QKV, first % QKV);
    if (P >= 1024) printf("  old %.3f ms  fused %.3f ms  (x%.2f)", ms_o, ms_n, ms_o / ms_n);
    printf("\n");
    if (mism) ok = false;
  }
  printf("CONVL2 PROTO: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
