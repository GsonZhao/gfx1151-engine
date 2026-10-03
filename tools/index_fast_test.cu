// index_fast_test.cu — bit-exact check + bench of the fast lightning-indexer
// kernels (src/gpu/parts/09_kernels_index.inc) against the production ones:
//   k_index_scores_t64  vs k_index_scores_tiled   (whole score matrix, memcmp)
//   k_index_select_2p   vs k_index_select_stream<true> (and _rs<true>, n<=8192)
// Select inputs: the real score rows from the score test, plus synthetic
// tie-heavy rows (quantized values, many zeros) to exercise the tie path,
// and cap = 0 / 64 to force the streaming fallback.
//   hipcc -O3 --offload-arch=gfx1151 -I src/gpu tools/index_fast_test.cu -o build/index_fast_test
//   build/index_fast_test [--iters N] [--quick]
// Ends with PASS or FAIL.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "parts/09_kernels_index.inc"

#define CK(x)                                                                    \
  do {                                                                           \
    hipError_t e_ = (x);                                                         \
    if (e_ != hipSuccess) {                                                      \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, \
              __LINE__);                                                         \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

static int g_fail = 0;

// C1 helpers: host bf16 (RNE) + exact top-512 (score desc, ties -> smaller id;
// same total order as the device selectors), used to measure the selection
// overlap between fp32 and bf16 score paths.
static uint16_t f2bf_host(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static float bf2f_host(uint16_t u) {
  uint32_t b = (uint32_t)u << 16;
  float f;
  memcpy(&f, &b, 4);
  return f;
}
// host f32 <-> f16 (RNE; manual bit ops so no device/__half host dependency)
static uint16_t f2h_host(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  int e = (int)((u >> 23) & 0xff) - 127 + 15;
  uint32_t m = u & 0x7fffffu;
  if (e >= 31) return (uint16_t)(sign | 0x7c00u);
  if (e <= 0) {
    if (e < -10) return (uint16_t)sign;
    m |= 0x800000u;
    const int sh = 14 - e;
    uint32_t h = m >> sh, rem = m & ((1u << sh) - 1);
    if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1))) h++;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)e << 10) | (m >> 13), rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
  return (uint16_t)(sign | h);
}
static float h2f_host(uint16_t h) {
  const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
  float v = e ? ldexpf((float)(1024 + m), (int)e - 25) : ldexpf((float)m, -24);
  return (h & 0x8000) ? -v : v;
}
template <class ST>
static void host_top512(const ST* row, int n, int* out) {
  std::vector<int> id(n);
  for (int i = 0; i < n; i++) id[i] = i;
  auto val = [&](int i) {
    if constexpr (std::is_same<ST, uint16_t>::value) return bf2f_host(row[i]);
    else return (float)row[i];
  };
  std::stable_sort(id.begin(), id.end(),
                   [&](int a, int b) { return val(a) > val(b); });
  const int m = std::min(n, 512);
  std::sort(id.begin(), id.begin() + m);
  for (int i = 0; i < 512; i++) out[i] = i < m ? id[i] : -1;
}
struct Case {
  const char* name;
  int nb, first, count;
};

// C1 shared analysis: value error + top-512 overlap of variant scores (hv,
// already converted to float) vs the fp32 reference (h0). Prints one line;
// sets g_fail when the mean overlap is below the 99.5% gate.
static void c1_analyze(const char* tag, const Case& c,
                       const std::vector<float>& h0,
                       const std::vector<float>& hv, bool sdiff) {
  double se = 0, sr = 0, sum_ov = 0, min_ov = 2;
  float maxd = 0;
  int nrows = 0, worst_t = -1;
  std::vector<int> sr_(512), sb_(512);
  for (int t = 0; t < c.count; t++) {
    const int lim = std::min(c.nb, (c.first + t + 1) / 4);
    for (int b = 0; b < lim; b++) {
      const float a = h0[(size_t)t * c.nb + b];
      const float v = hv[(size_t)t * c.nb + b];
      const float d = fabsf(a - v);
      if (d > maxd) maxd = d;
      se += (double)d * d;
      sr += (double)a * a;
    }
    if (lim > 512) {
      host_top512(&h0[(size_t)t * c.nb], lim, sr_.data());
      host_top512(&hv[(size_t)t * c.nb], lim, sb_.data());
      int i = 0, j = 0, ov = 0;  // both ascending: merge-intersect
      while (i < 512 && j < 512) {
        if (sr_[i] == sb_[j]) { ov++; i++; j++; }
        else if (sr_[i] < sb_[j]) i++;
        else j++;
      }
      const double o = ov / 512.0;
      sum_ov += o;
      nrows++;
      if (o < min_ov) { min_ov = o; worst_t = t; }
    }
  }
  printf("  %-10s %-28s maxdiff=%.4g relL2=%.3g  overlap mean=%.5f min=%.4f (rows=%d, worst t=%d)%s\n",
         tag, c.name, maxd, std::sqrt(se / std::max(sr, 1e-30)),
         nrows ? sum_ov / nrows : 1.0, nrows ? min_ov : 1.0, nrows, worst_t,
         sdiff ? "  DEVICE-SELECT-DIFF" : "");
  if (sdiff) g_fail = 1;
  if (nrows && sum_ov / nrows < 0.995) {
    printf("  [%s %s] overlap below the 99.5%% gate\n", tag, c.name);
    g_fail = 1;
  }
}

static double flops_of(const Case& c) {
  double f = 0;
  for (int t = 0; t < c.count; t++) f += std::min(c.nb, (c.first + t + 1) / 4);
  return f * 4 * 128 * 2;
}

template <class F>
static float time_ms(F&& f, int iters) {
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  f();
  CK(hipDeviceSynchronize());
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; i++) f();
  CK(hipEventRecord(b));
  CK(hipEventSynchronize(b));
  float ms;
  CK(hipEventElapsedTime(&ms, a, b));
  return ms / iters;
}

// compare the two selectors on a score matrix already on the device
static void check_select(const char* tag, const float* d_sc, int stride, int first, int count,
                         int* d_s0, int* d_s1, bool with_rs) {
  size_t nsel = (size_t)count * 512;
  std::vector<int> h0(nsel), h1(nsel);
  CK(hipMemset(d_s0, 0x7f, nsel * 4));
  k_index_select_stream<true><<<count, 256>>>(d_sc, stride, d_s0, first);
  CK(hipGetLastError());
  CK(hipMemcpy(h0.data(), d_s0, nsel * 4, hipMemcpyDeviceToHost));
  // sanity: ascending, in range
  for (int q = 0; q < count; q++) {
    int n = (first + q + 1) / 4;
    for (int i = 1; i < 512 && i < n; i++)
      if (h0[(size_t)q * 512 + i] <= h0[(size_t)q * 512 + i - 1]) {
        printf("  [%s] reference not ascending at q=%d i=%d\n", tag, q, i);
        g_fail = 1;
        q = count;
        break;
      }
  }
  if (with_rs) {
    CK(hipMemset(d_s1, 0x7f, nsel * 4));
    k_index_select_rs<true><<<count, 256>>>(d_sc, stride, d_s1, first);
    CK(hipMemcpy(h1.data(), d_s1, nsel * 4, hipMemcpyDeviceToHost));
    if (memcmp(h0.data(), h1.data(), nsel * 4)) {
      printf("  [%s] note: old rs != old stream (reference disagreement)\n", tag);
      g_fail = 1;
    }
  }
  const int caps[3] = {ISEL_CAP, 64, 0};
  for (int ci = 0; ci < 3; ci++) {
    CK(hipMemset(d_s1, 0x7f, nsel * 4));
    k_index_select_2p<<<count, 256>>>(d_sc, stride, d_s1, first, caps[ci]);
    CK(hipGetLastError());
    CK(hipMemcpy(h1.data(), d_s1, nsel * 4, hipMemcpyDeviceToHost));
    size_t diff = 0, firstd = 0;
    for (size_t i = 0; i < nsel; i++)
      if (h0[i] != h1[i]) {
        if (!diff) firstd = i;
        diff++;
      }
    printf("  [%s] select_2p cap=%-4d vs stream: %s", tag, caps[ci], diff ? "DIFF" : "identical");
    if (diff)
      printf(" (%zu ids, first q=%zu i=%zu: %d vs %d)", diff, firstd / 512, firstd % 512,
             h0[firstd], h1[firstd]);
    printf("\n");
    if (diff) g_fail = 1;
  }
}

int main(int argc, char** argv) {
  int iters = 20;
  bool quick = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
    else if (a == "--quick") quick = true;
  }
  const Case cases[] = {
      {"8K chunk, first batch", 2048, 2051, 1024},
      {"8K chunk, ragged tail", 2048, 7171, 1021},
      {"small count 17", 4099, 16000, 17},
      {"count 16", 4099, 16383, 16},
      {"early batch of a 48K chunk", 12048, 40000, 1024},
      {"n crosses 8192 (32K)", 8704, 32000, 1024},
      {"odd nb 30001", 30001, 119000, 1000},
      {"128K last batch", 32768, 130048, 1024},
      {"256K last batch", 65536, 261120, 1024},
  };
  const int ncases = sizeof(cases) / sizeof(cases[0]);
  size_t max_q = 1024 * 512, max_k = 65536 * 128, max_sc = 1024 * (size_t)65536;
  float *d_q, *d_k, *d_s0, *d_s1;
  uint16_t *d_qb, *d_kb, *d_sb, *d_qh, *d_kh, *d_sh;
  int *d_i0, *d_i1;
  CK(hipMalloc(&d_q, max_q * 4));
  CK(hipMalloc(&d_k, max_k * 4));
  CK(hipMalloc(&d_s0, max_sc * 4));
  CK(hipMalloc(&d_s1, max_sc * 4));
  CK(hipMalloc(&d_qb, max_q * 2));
  CK(hipMalloc(&d_kb, max_k * 2));
  CK(hipMalloc(&d_sb, max_sc * 2));
  CK(hipMalloc(&d_qh, max_q * 2));
  CK(hipMalloc(&d_kh, max_k * 2));
  CK(hipMalloc(&d_sh, max_sc * 2));
  CK(hipMalloc(&d_i0, 1024 * 512 * 4));
  CK(hipMalloc(&d_i1, 1024 * 512 * 4));
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> hq(max_q), hk(max_k);
  {
    // indexer q/k are RMS-normed (+rope) fp32 vectors; unit-normal entries
    // with a per-row scale spread give a realistic, tie-free score spread.
    for (size_t r = 0; r < max_q / 128; r++) {
      float sc = 0.5f + (float)(r % 7) * 0.1f;
      for (int d = 0; d < 128; d++) hq[r * 128 + d] = nd(rng) * sc * 0.25f;
    }
    for (size_t r = 0; r < max_k / 128; r++) {
      float sc = 0.6f + (float)(r % 5) * 0.15f;
      for (int d = 0; d < 128; d++) hk[r * 128 + d] = nd(rng) * sc;
    }
    CK(hipMemcpy(d_q, hq.data(), max_q * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_k, hk.data(), max_k * 4, hipMemcpyHostToDevice));
    std::vector<uint16_t> hqb(max_q), hkb(max_k);
    for (size_t i = 0; i < max_q; i++) hqb[i] = f2bf_host(hq[i]);
    for (size_t i = 0; i < max_k; i++) hkb[i] = f2bf_host(hk[i]);
    CK(hipMemcpy(d_qb, hqb.data(), max_q * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_kb, hkb.data(), max_k * 2, hipMemcpyHostToDevice));
    std::vector<uint16_t> hqh(max_q), hkh(max_k);
    for (size_t i = 0; i < max_q; i++) hqh[i] = f2h_host(hq[i]);
    for (size_t i = 0; i < max_k; i++) hkh[i] = f2h_host(hk[i]);
    CK(hipMemcpy(d_qh, hqh.data(), max_q * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_kh, hkh.data(), max_k * 2, hipMemcpyHostToDevice));
  }
  printf("== scores: k_index_scores_t64 vs k_index_scores_tiled (full matrix, bitwise)\n");
  for (int ci = 0; ci < ncases; ci++) {
    const Case& c = cases[ci];
    if (quick && c.nb > 32768) continue;
    size_t n = (size_t)c.count * c.nb;
    CK(hipMemset(d_s0, 0xff, n * 4));
    CK(hipMemset(d_s1, 0xff, n * 4));
    k_index_scores_tiled<<<dim3((c.nb + 31) / 32, (c.count + 15) / 16), 256>>>(
        d_q, d_k, d_s0, c.nb, c.count, c.first);
    CK(hipGetLastError());
    k_index_scores_t64<<<index_t64_grid(c.nb, c.count), 256>>>(
        d_q, d_k, d_s1, c.nb, c.count, c.first);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    std::vector<uint32_t> h0(n), h1(n);
    CK(hipMemcpy(h0.data(), d_s0, n * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(h1.data(), d_s1, n * 4, hipMemcpyDeviceToHost));
    size_t diff = 0, fd = 0, written = 0;
    for (size_t i = 0; i < n; i++) {
      if (h0[i] != 0xffffffffu) written++;
      if (h0[i] != h1[i]) {
        if (!diff) fd = i;
        diff++;
      }
    }
    printf("  %-28s nb=%-6d first=%-6d count=%-5d written=%zu  %s", c.name, c.nb, c.first,
           c.count, written, diff ? "DIFF" : "identical");
    if (diff) {
      float a, b;
      memcpy(&a, &h0[fd], 4);
      memcpy(&b, &h1[fd], 4);
      printf(" (%zu, first t=%zu b=%zu: %.9g vs %.9g)", diff, fd / c.nb, fd % c.nb, a, b);
      g_fail = 1;
    }
    printf("\n");
    // selection on these real scores (only meaningful when the rows are long)
    if ((c.first + c.count) / 4 > 512)
      check_select(c.name, d_s0, c.nb, c.first, c.count, d_i0, d_i1,
                   (c.first + c.count) / 4 <= 8192);
  }

  printf("== C1: k_index_scores_bf16 (bf16 WMMA) vs t64 fp32; overlap + device select\n");
  for (int ci = 0; ci < ncases; ci++) {
    const Case& c = cases[ci];
    if (quick && c.nb > 32768) continue;
    size_t n = (size_t)c.count * c.nb;
    CK(hipMemset(d_s0, 0xff, n * 4));
    CK(hipMemset(d_sb, 0xff, n * 2));
    k_index_scores_t64<<<index_t64_grid(c.nb, c.count), 256>>>(
        d_q, d_k, d_s0, c.nb, c.count, c.first);
    CK(hipGetLastError());
    k_index_scores_wmma16<true><<<index_wmma16_grid(c.nb, c.count), 256>>>(
        d_qb, d_kb, d_sb, c.nb, c.count, c.first);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    std::vector<float> h0(n);
    std::vector<uint16_t> hb(n);
    CK(hipMemcpy(h0.data(), d_s0, n * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hb.data(), d_sb, n * 2, hipMemcpyDeviceToHost));
    // device select on the bf16 scores, exactly as the engine would route it
    const bool use_rs = (c.first + c.count) / 4 <= 8192;
    CK(hipMemset(d_i1, 0x7f, (size_t)c.count * 512 * 4));
    if (use_rs)
      k_index_select_rs<true, uint16_t><<<c.count, 256>>>(d_sb, c.nb, d_i1, c.first);
    else
      k_index_select_2p<uint16_t><<<c.count, 256>>>(d_sb, c.nb, d_i1, c.first);
    CK(hipGetLastError());
    std::vector<int> hid((size_t)c.count * 512);
    CK(hipMemcpy(hid.data(), d_i1, hid.size() * 4, hipMemcpyDeviceToHost));
    std::vector<float> hv(n);
    std::vector<int> sb_(512);
    int sdiff = 0;
    for (size_t i = 0; i < n; i++) hv[i] = bf2f_host(hb[i]);
    for (int t = 0; t < c.count; t++) {
      const int lim = std::min(c.nb, (c.first + t + 1) / 4);
      host_top512(&hb[(size_t)t * c.nb], lim, sb_.data());
      for (int i = 0; i < 512; i++)
        if (sb_[i] != hid[(size_t)t * 512 + i]) sdiff = 1;
    }
    c1_analyze("bf16", c, h0, hv, sdiff);
  }

  printf("== C1-f16: k_index_scores_wmma16<false> (f16 WMMA) vs t64 fp32\n");
  for (int ci = 0; ci < ncases; ci++) {
    const Case& c = cases[ci];
    if (quick && c.nb > 32768) continue;
    size_t n = (size_t)c.count * c.nb;
    CK(hipMemset(d_s0, 0xff, n * 4));
    CK(hipMemset(d_sh, 0xff, n * 2));
    k_index_scores_t64<<<index_t64_grid(c.nb, c.count), 256>>>(
        d_q, d_k, d_s0, c.nb, c.count, c.first);
    CK(hipGetLastError());
    k_index_scores_wmma16<false><<<index_wmma16_grid(c.nb, c.count), 256>>>(
        d_qh, d_kh, d_sh, c.nb, c.count, c.first);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    std::vector<float> h0(n);
    std::vector<uint16_t> hb(n);
    CK(hipMemcpy(h0.data(), d_s0, n * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hb.data(), d_sh, n * 2, hipMemcpyDeviceToHost));
    // device select on the f16 scores, exactly as the engine would route it
    const bool use_rs = (c.first + c.count) / 4 <= 8192;
    CK(hipMemset(d_i1, 0x7f, (size_t)c.count * 512 * 4));
    if (use_rs)
      k_index_select_rs<true, __half><<<c.count, 256>>>((const __half*)d_sh, c.nb,
                                                       d_i1, c.first);
    else
      k_index_select_2p<__half><<<c.count, 256>>>((const __half*)d_sh, c.nb, d_i1,
                                                  c.first);
    CK(hipGetLastError());
    std::vector<int> hid((size_t)c.count * 512);
    CK(hipMemcpy(hid.data(), d_i1, hid.size() * 4, hipMemcpyDeviceToHost));
    std::vector<float> hv(n);
    std::vector<int> sb_(512);
    int sdiff = 0;
    for (size_t i = 0; i < n; i++) hv[i] = h2f_host(hb[i]);
    for (int t = 0; t < c.count; t++) {
      const int lim = std::min(c.nb, (c.first + t + 1) / 4);
      host_top512(&hv[(size_t)t * c.nb], lim, sb_.data());
      for (int i = 0; i < 512; i++)
        if (sb_[i] != hid[(size_t)t * 512 + i]) sdiff = 1;
    }
    c1_analyze("f16", c, h0, hv, sdiff);
  }


  printf("== select: synthetic tie-heavy rows\n");
  {
    const int nb = 32768, first = 130048, count = 256;
    std::vector<float> hs((size_t)count * nb);
    for (int q = 0; q < count; q++)
      for (int b = 0; b < nb; b++) {
        float v;
        switch (q % 4) {
          case 0: v = (float)(rng() % 40) * 0.125f; break;           // huge ties
          case 1: v = (rng() % 100 < 99) ? 0.f : (float)(rng() % 7); break;  // mostly 0
          case 2: v = (rng() % 3 == 0) ? 1.5f : 1.5f - 0.001f * (rng() % 5); break;
          default: v = std::fabs(nd(rng)) * 3.f; break;
        }
        hs[(size_t)q * nb + b] = v;
      }
    CK(hipMemcpy(d_s0, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
    check_select("ties 128K", d_s0, nb, first, count, d_i0, d_i1, false);
    check_select("ties 30K", d_s0, nb, 30000, count, d_i0, d_i1, true);
  }

  printf("== speed (128K / 256K last batch, %d iters)\n", iters);
  for (int ci = 0; ci < ncases; ci++) {
    const Case& c = cases[ci];
    if (c.nb < 32768 || (quick && c.nb > 32768)) continue;
    auto told = [&] {
      k_index_scores_tiled<<<dim3((c.nb + 31) / 32, (c.count + 15) / 16), 256>>>(
          d_q, d_k, d_s0, c.nb, c.count, c.first);
    };
    auto tnew = [&] {
      k_index_scores_t64<<<index_t64_grid(c.nb, c.count), 256>>>(
          d_q, d_k, d_s1, c.nb, c.count, c.first);
    };
    auto tbf = [&] {
      k_index_scores_wmma16<true><<<index_wmma16_grid(c.nb, c.count), 256>>>(
          d_qb, d_kb, d_sb, c.nb, c.count, c.first);
    };
    auto tfh = [&] {
      k_index_scores_wmma16<false><<<index_wmma16_grid(c.nb, c.count), 256>>>(
          d_qh, d_kh, d_sh, c.nb, c.count, c.first);
    };
    float mo = time_ms(told, iters), mn = time_ms(tnew, iters),
          mb = time_ms(tbf, iters), mh = time_ms(tfh, iters);
    double f = flops_of(c);
    printf("  %-18s score  old %7.3f ms (%5.2f TFLOPS)  new %7.3f ms (%5.2f TFLOPS)  x%.2f\n",
           c.name, mo, f / mo / 1e9, mn, f / mn / 1e9, mo / mn);
    printf("  %-18s score  bf16 %6.3f ms (%5.2f TF) x%.2f  f16 %6.3f ms (%5.2f TF) x%.2f vs t64\n",
           c.name, mb, f / mb / 1e9, mn / mb, mh, f / mh / 1e9, mn / mh);
    auto sold = [&] {
      k_index_select_stream<true><<<c.count, 256>>>(d_s0, c.nb, d_i0, c.first);
    };
    auto snew = [&] { k_index_select_2p<<<c.count, 256>>>(d_s0, c.nb, d_i1, c.first); };
    auto sbf = [&] {
      k_index_select_2p<uint16_t><<<c.count, 256>>>(d_sb, c.nb, d_i1, c.first);
    };
    auto sfh = [&] {
      k_index_select_2p<__half><<<c.count, 256>>>((const __half*)d_sh, c.nb, d_i1,
                                                  c.first);
    };
    float so = time_ms(sold, iters), sn = time_ms(snew, iters),
          sb2 = time_ms(sbf, iters), sh2 = time_ms(sfh, iters);
    printf("  %-18s select old %7.3f ms  new %7.3f ms  x%.2f  bf16 %7.3f ms  x%.2f  f16 %7.3f ms  x%.2f\n",
           c.name, so, sn, so / sn, sb2, sn / sb2, sh2, sn / sh2);
  }
  CK(hipDeviceSynchronize());
  printf(g_fail ? "FAIL\n" : "PASS\n");
  return g_fail;
}
