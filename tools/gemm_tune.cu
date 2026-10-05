// gemm_tune.cu — k_gemm_wmma tile-config sweep for the narrow prefill shapes
// (aggr 10-05). Includes the engine kernel verbatim (sed-extracted):
//   sed -n '/\[gemm-wmma-begin\]/,/\[gemm-wmma-end\]/p' src/gpu/parts/22_kernels_prefill.inc > /tmp/gemm_wmma_kernel_live.inc
//   hipcc -O3 --offload-arch=gfx1151 -I/tmp -o build/gemm_tune tools/gemm_tune.cu
//   build/gemm_tune <shape> <P>    shape: oproj | mixdn | mixdn4 | upfused | all
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                    \
  do {                                                                           \
    hipError_t e_ = (x);                                                         \
    if (e_ != hipSuccess) {                                                      \
      fprintf(stderr, "hip %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
      exit(1);                                                                   \
    }                                                                            \
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
#include "gemm_wmma_kernel_live.inc"

static uint16_t *dX, *dW, *dR, *dYb;
static float *dY, *dSc, *dHw;

template <int BM, int BP, int MW, int PW, int WM, int WP, int KST, int NT, int Epi = 0>
static void run(const char* tag, int P, int K, int N, int gm, int split = 1, bool sc = false) {
  constexpr int LDS = 2 * (BM * (KST + 8) + (BP / 8) * (8 * (KST + 8) + 8)) * 2;
  if (LDS > 65536) return;
  const int ks = K / split;
  if (ks % KST) return;
  const unsigned gx = (unsigned)(((P + BM - 1) / BM) * ((N + BP - 1) / BP));
  dim3 grid(gx, split);
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  auto launch = [&] {
    if (split == 1)
      k_gemm_wmma<BM, BP, MW, PW, WM, WP, KST, NT, Epi><<<grid, NT>>>(
          dX, dW, dY, P, K, N, gm, dR, dYb, sc ? dSc : nullptr, sc ? dHw : nullptr);
    else
      k_gemm_wmma<BM, BP, MW, PW, WM, WP, KST, NT, Epi><<<grid, NT>>>(
          dX, dW, dY, P, ks, N, gm, dR, dYb, nullptr, nullptr, K, K, ks, ks,
          (size_t)P * N);
  };
  for (int i = 0; i < 3; i++) launch();
  CK(hipDeviceSynchronize());
  const int it = 20;
  CK(hipEventRecord(a));
  for (int i = 0; i < it; i++) launch();
  CK(hipEventRecord(b));
  CK(hipEventSynchronize(b));
  float ms = 0;
  CK(hipEventElapsedTime(&ms, a, b));
  ms /= it;
  const double tf = 2.0 * P * K * N / (ms * 1e-3) / 1e12;
  printf("%-10s P=%-6d <%3d,%3d,%d,%d,%d,%d,%3d,%3d,E%d> gm=%-2d split=%d LDS=%5d  %7.3f ms  %5.1f TF\n",
         tag, P, BM, BP, MW, PW, WM, WP, KST, NT, Epi, gm, split, LDS, ms, tf);
  CK(hipEventDestroy(a));
  CK(hipEventDestroy(b));
}

// generic Epi=0 sweep for (N, K)
static void sweep0(const char* tag, int P, int K, int N, int split = 1) {
  for (int gm : {1, 4, 16}) {
    run<64, 128, 2, 2, 2, 4, 64, 128>(tag, P, K, N, gm, split);
    run<64, 128, 2, 2, 2, 4, 32, 128>(tag, P, K, N, gm, split);
    run<128, 128, 2, 2, 4, 4, 32, 128>(tag, P, K, N, gm, split);
    run<128, 128, 4, 2, 2, 4, 32, 256>(tag, P, K, N, gm, split);
    run<128, 128, 2, 4, 4, 2, 32, 256>(tag, P, K, N, gm, split);
    run<128, 256, 2, 4, 4, 4, 32, 256>(tag, P, K, N, gm, split);
    run<128, 256, 2, 8, 4, 2, 32, 512>(tag, P, K, N, gm, split);
    run<256, 128, 4, 2, 4, 4, 32, 256>(tag, P, K, N, gm, split);
    run<256, 128, 8, 2, 2, 4, 32, 512>(tag, P, K, N, gm, split);
    run<128, 64, 4, 1, 2, 4, 64, 128>(tag, P, K, N, gm, split);
    run<128, 64, 4, 2, 2, 2, 64, 256>(tag, P, K, N, gm, split);
  }
}
static void sweep_mixdn(const char* tag, int P, int split) {
  const int K = 10240, N = 320;
  for (int gm : {1, 2}) {
    run<64, 160, 2, 2, 2, 5, 64, 128>(tag, P, K, N, gm, split);  // engine
    run<64, 160, 2, 2, 2, 5, 32, 128>(tag, P, K, N, gm, split);
    run<32, 160, 1, 2, 2, 5, 64, 64>(tag, P, K, N, gm, split);
    run<32, 160, 1, 2, 2, 5, 32, 64>(tag, P, K, N, gm, split);
    run<64, 80, 4, 1, 1, 5, 64, 128>(tag, P, K, N, gm, split);
    run<64, 320, 1, 4, 4, 5, 32, 128>(tag, P, K, N, gm, split);
  }
}
static void sweep_up(const char* tag, int P) {
  const int K = 320, N = 10240;
  for (int gm : {1, 2, 4}) {
    run<128, 128, 4, 2, 2, 4, 32, 256, 1>(tag, P, K, N, gm, 1, true);  // engine
    run<64, 128, 2, 2, 2, 4, 32, 128, 1>(tag, P, K, N, gm, 1, true);
    run<128, 128, 2, 2, 4, 4, 32, 128, 1>(tag, P, K, N, gm, 1, true);
    run<256, 128, 4, 2, 4, 4, 32, 256, 1>(tag, P, K, N, gm, 1, true);
    run<128, 256, 2, 4, 4, 4, 32, 256, 1>(tag, P, K, N, gm, 1, true);
    run<128, 256, 4, 4, 2, 4, 32, 512, 1>(tag, P, K, N, gm, 1, true);
    run<64, 256, 1, 4, 4, 4, 32, 128, 1>(tag, P, K, N, gm, 1, true);
    run<256, 64, 8, 1, 2, 4, 32, 256, 1>(tag, P, K, N, gm, 1, true);
  }
}

int main(int argc, char** argv) {
  const std::string shape = argc > 1 ? argv[1] : "all";
  const int P = argc > 2 ? atoi(argv[2]) : 16384;
  const size_t nX = (size_t)P * 10240, nW = (size_t)10240 * 6144;
  CK(hipMalloc(&dX, nX * 2));
  CK(hipMalloc(&dW, nW * 2));
  CK(hipMalloc(&dR, (size_t)P * 10240 * 2));
  CK(hipMalloc(&dYb, (size_t)P * 2560 * 2));
  CK(hipMalloc(&dY, (size_t)P * 10240 * 4));
  CK(hipMalloc(&dSc, (size_t)P * 4 * 4));
  CK(hipMalloc(&dHw, 10240 * 4));
  {
    std::vector<uint16_t> h(nW);
    for (size_t i = 0; i < nW; i++) h[i] = 0x3C00 ^ (uint16_t)((i * 2654435761u) >> 20 & 0x3FF);
    CK(hipMemcpy(dW, h.data(), nW * 2, hipMemcpyHostToDevice));
    h.resize(nX);
    for (size_t i = 0; i < nX; i++) h[i] = 0x3C00 ^ (uint16_t)((i * 40503u) >> 6 & 0x3FF);
    CK(hipMemcpy(dX, h.data(), nX * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dR, h.data(), (size_t)P * 10240 * 2, hipMemcpyHostToDevice));
    CK(hipMemset(dSc, 0, (size_t)P * 16));
    CK(hipMemset(dHw, 0, 10240 * 4));
  }
  if (shape == "oproj" || shape == "all") {
    sweep0("oproj", P, 6144, 2560);
    sweep0("oproj/s2", P, 6144, 2560, 2);
  }
  if (shape == "mixdn" || shape == "all") sweep_mixdn("mixdn", P, 1);
  if (shape == "mixdn4" || shape == "all") sweep_mixdn("mixdn/s4", P, 4);
  if (shape == "upfused" || shape == "all") sweep_up("upfused", P);
  if (shape == "d9") {
    sweep0("qkv", P, 2560, 10240);
    sweep0("z", P, 2560, 6144);
  }
  return 0;
}
