# gfx1151-engine

*中文版:[README.md](README.md)*

A local inference engine that runs a 177B MoE model on a single AMD Strix
Halo APU (gfx1151). Target model: Qwen3.8-Flash-Next (qwen4_exp architecture)
and fine-tunes with the same architecture.

The routed experts are stored in host memory in 4-bit quantization and read
directly by the GPU kernels — no large VRAM needed. The memory-resident
weights are ~67–77 GiB depending on the weight format (the PLE n-gram table
stays on disk and is read on demand); a machine with 122 GiB of RAM can serve
a 256K context.

## Measured Performance

Development machine: **GMK EVO-X2 (AMD Ryzen AI Max+ 395, Strix Halo /
gfx1151)**, 122 GiB RAM:

| Metric | Value |
| --- | --- |
| Prefill (128K context) | ~1400–1470 tok/s |
| Decode (speculative) | ~54–55 tok/s at 8K/32K context, ~36–45 tok/s at 64K (depends on weight format) |
| Decode (no speculation) | ~25–30 tok/s (depends on weight format) |
| Average power draw | ~120 W |
| Peak (instantaneous) power draw | ~130 W (bursts for a few seconds, then settles back to ~120 W) |

## Weight Formats and Quality

The engine supports two weight formats. Service, API and speculative
decoding are identical; switch by using the other launcher:

| | hgn standard | hgn high quality (HQ) | GGUF UD-Q4_K_XL |
| --- | --- | --- | --- |
| Files | current default (`qwen38-flash-next-w4b.hgn` + overlay) | converted from the original weights, see [HGN-HQ.md](HGN-HQ.md) (Chinese) | released by Unsloth, the same files llama.cpp uses |
| Routed experts | 4-bit (q4cp) | 4-bit (q4cp, imatrix-weighted) | mostly Q4_K / Q5_1 |
| Dense (attention, GDN, shared expert, embed, lm_head) | 4-bit | 8-bit (q8g32 overlay) | 8-bit (Q8_0) |
| KLD vs BF16 (lower is better) | 0.163 | **0.0558** | 0.0511 |
| top1 agreement with BF16 | 86.8% | 92.4% | 92.6% |
| Resident weights bpw / size | 4.55 / 66.6 GiB | 4.70 / 68.8 GiB | 5.25 / 76.9 GiB |
| Prefill (8K prompt, chunk 2048) | ~1200 tok/s | ~1200 tok/s | ~1200 tok/s |
| Decode (no speculation) | ~30 tok/s | ~25 tok/s | ~25 tok/s |
| Launch | `start_hgn.sh` | `start_hgn.sh` (set `MODEL_FILE` / `OVERLAY_FILE`) | `start_gguf.sh` |
| Windows | yes | yes (not yet measured) | no |

- KLD: BF16 reference, wikitext-2, 64 chunks × 512, measured the same way as
  unsloth / llama.cpp (see [KLD.md](KLD.md), Chinese); llama.cpp on the same
  GGUF gives 0.049.
- Almost the whole quality gap comes from the dense bit width: 8-bit dense
  takes KLD from 0.163 to 0.063, imatrix-weighted experts bring it to 0.0558.
  The cost is more bytes read per decode token, ~16% slower decode (same as
  GGUF); prefill is unaffected.
- Resident weights exclude the PLE n-gram table (hgn fp8 47.7 GiB, GGUF
  IQ4_NL 26.8 GiB), which stays on disk and is read on demand.
  `python3 tools/bpw.py` reports bpw per category for each file (reads headers
  only, a few seconds).
- The default configuration is still hgn standard. The HQ files pass
  `tools/hq_verify.sh`; deployment is described in section 6 of
  [HGN-HQ.md](HGN-HQ.md).

## Features

- **Speculative decoding chain**: ngram drafts first, MTP as fallback, with
  round-by-round fallback; greedy does bit-exact comparison, sampling
  resamples and compares against the target distribution, so the output
  distribution is identical to serial decoding. Enabled by default, no
  parameters needed. Highly repetitive content (code comments, template
  text) shows significant measured speedups.
- **Standalone 8-bit MTP draft weights** sidecar, with a higher acceptance
  rate than the built-in 4-bit draft head.
- **Vision**: supports image input (OpenAI `image_url`), with KV reuse
  across turns.
- **OpenAI-compatible API**: streaming, tool calling,
  `/v1/chat/completions`.
- **256K context**, with two-tier prompt caching: in-RAM checkpoints at
  message boundaries (edit-and-resend replies instantly) plus KV snapshots
  recovered across restarts.
- **Two weight formats**: the native `.hgn` (Linux / Windows) and llama.cpp
  GGUF (Unsloth UD-Q4_K_XL, Linux); comparison above.
- **Model conversion tool**: HF safetensors → `.hgn`. The default output is
  the high-quality variant (8-bit dense overlay + weighted 4-bit experts); a
  llama.cpp-format imatrix is used if you have one, and conversion works
  without one too. Usable for your own fine-tunes of the same architecture
  (see [HGN-HQ.md](HGN-HQ.md), [CONVERT_EN.md](CONVERT_EN.md)).

## Requirements

- Linux + ROCm (HIP 7.x), GPU architecture `gfx1151`; or Windows + AMD GPU
  driver (the GPU needs VRAM carved out in BIOS), see the "Windows" section
- Available memory ≥ 100 GiB (pinned (page-locked) weights: ~68 GiB for hgn,
  ~80 GiB for GGUF, plus KV)
- Build dependencies: rocBLAS, hipBLASLt, rocPRIM; the API frontend also
  needs libpng, libjpeg, libwebp; nlohmann/json is vendored in the repository

## Quick Start

```bash
bash build.sh        # Build engine + API, output goes to build/
bash start_hgn.sh    # hgn weights: load the model and start the service (reads service.conf)
bash start_gguf.sh   # or GGUF weights (Unsloth UD-Q4_K_XL, the same files llama.cpp uses)
```

Both launchers read weights from `./models` and list any missing files before
exiting; configuration is centralized in `service.conf` (one section each for
hgn and GGUF; GGUF details in [GGUF.md](GGUF.md)). See [QUICKSTART_EN.md](QUICKSTART_EN.md)
for details.

Converting your own fine-tuned model (HF safetensors, same architecture):

```bash
# without an imatrix
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models
# with an imatrix (llama.cpp format: GGUF or legacy imatrix.dat)
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models --imatrix /path/to/imatrix.gguf
```

It writes the base `.hgn`, the 8-bit dense overlay, the 8-bit MTP draft, the
vision tower, the tokenizer and a ready-to-run `start.sh`; ~1.5 hours on 32
cores, ~125 GiB of disk, needs only numpy. `--classic` is the old data-free
converter (byte-identical output to before). High-quality conversion: see
[HGN-HQ.md](HGN-HQ.md); format and the old converter: see
[CONVERT_EN.md](CONVERT_EN.md).

## Windows

The Windows version has feature parity with the Linux version (engine +
OpenAI API + multimodal). Porting notes and measurements are in
[PORTING-WINDOWS_EN.md](PORTING-WINDOWS_EN.md). Builds run in Git Bash
(or double-click `build_win.bat`; Git is only needed at build time):

```bash
bash build_win.sh           # Engine
bash build_win.sh api       # OpenAI API frontend
bash build_win.sh launcher  # Script-free launcher start_win.exe
```

For daily use, double-click `start_win.exe` (native Win32, no
Git/PowerShell needed): it brings up the engine + API dual processes,
shows output in real time and writes it to `logs\`; Ctrl+C or closing the
window stops it. Configuration is **shared with Linux via
`service.conf`** (edit it to change the model file name or context
window); environment variables can temporarily override it. Clients
connect to `http://<host>:8731/v1`.

Distribution: copy `build/` + `start_win.exe` + `models/` to any gfx1151
Windows machine and it just works — **no ROCm/TheRock installation
needed**; only the AMD GPU driver, plus enough VRAM carved out for the GPU
in BIOS (a 256K context needs 96 GiB).

Differences from the Linux version:

- hgn weights only (GGUF is not ported). The high-quality hgn works by
  swapping files: weight arena +2.2 GiB, estimated ~93.2 GiB at 256K /
  chunk 8192 (limit 95); not yet measured on Windows
- Image decoding supports PNG/JPEG via stb_image (WebP not wired up)
- Prefill chunk defaults to 8192
- Cold loading reads the full weights from disk (minute-scale, progress
  shown in console/logs)
- The launchers do not enable `GDEC_GEMM_WMMA` or `GDEC_GDN_FUSED` (the
  self-written WMMA GEMM and fused GDN kernel already promoted on Linux
  launchers, worth ~8-10% PP combined but unverified under TheRock — so
  Windows prefill uses hipBLASLt plus the legacy GDN path)

Build details are in [BUILD_EN.md](BUILD_EN.md).

## Documentation

- [QUICKSTART_EN.md](QUICKSTART_EN.md) — build, launch, configuration
- [BUILD_EN.md](BUILD_EN.md) — build environment details and
  troubleshooting
- [GGUF.md](GGUF.md) (Chinese) — GGUF weight loading, performance vs hgn
- [HGN-HQ.md](HGN-HQ.md) (Chinese) — high-quality hgn: one-step conversion
  (optional imatrix), results, deployment
- [CONVERT_EN.md](CONVERT_EN.md) — model conversion tool
- [KLD.md](KLD.md) (Chinese) — quality testing (KLD, same method as
  unsloth / llama.cpp)
- [MTP_EN.md](MTP_EN.md) — speculative decoding parameters and comparison
  methods
- [NGRAM_EN.md](NGRAM_EN.md) — ngram verification design, benefits, and
  known divergences
- [HGN-FORMAT_EN.md](HGN-FORMAT_EN.md) — the `.hgn` weight container
  format
- [data/README_EN.md](data/README_EN.md) — numerical regression benchmark
  (data/qsa-oracle) description
- [PORTING-WINDOWS_EN.md](PORTING-WINDOWS_EN.md) — Windows porting notes
  and measurements

## Tests

```bash
bash build.sh test     # Kernel unit tests, no model loading, expect ALL PASS
python3 tools/bpw.py   # bpw of the weights under models/ by category (headers only)
```

Quality (KLD) testing needs a BF16 reference; see [KLD.md](KLD.md).

## Acknowledgements

This project's implementation borrows from [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) by peonist-ai. The `.hgn` weight container format is halogen's checkpoint container format — see [HGN-FORMAT_EN.md](HGN-FORMAT_EN.md). Many thanks to the halogen authors.

The ngram speculative drafting approach also borrows ideas from the open-source [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT license); see the "Attribution" section of [NGRAM_EN.md](NGRAM_EN.md). The GGUF routed-expert WMMA kernel is ported from gufo (MIT license), see [GGUF.md](GGUF.md).

## License

[AGPL-3.0](LICENSE)
