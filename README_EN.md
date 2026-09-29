# gfx1151-engine

*中文版:[README.md](README.md)*

A local inference engine that runs a 177B MoE model on a single AMD Strix
Halo APU (gfx1151). Target model: Qwen3.8-Flash-Next (qwen4_exp architecture)
and fine-tunes with the same architecture.

The 68 GiB of quantized weights are stored in host memory in 4-bit
quantization and read directly by the GPU kernels — no large VRAM needed;
a machine with 122 GiB of RAM can serve a 256K context.

## Measured Performance

Development machine: **GMK EVO-X2 (AMD Ryzen AI Max+ 395, Strix Halo /
gfx1151)**, 122 GiB RAM:

| Metric | Value |
| --- | --- |
| Prefill (128K context) | ~1400–1470 tok/s (measured 09-28) |
| Decode (speculative, greedy, γ=4) | ~45 tok/s at 8K, ~46 at 64K (real text, 09-29) |
| Decode (speculative, sampling, adaptive γ) | ~40 tok/s at 8K, ~45 at 64K (real text, 09-29) |
| Decode (no speculation) | ~25–30 tok/s (measured 09-28, depends on weight format) |
| Average power draw | ~120 W |
| Peak (instantaneous) power draw | ~130 W (bursts for a few seconds, then settles back to ~120 W) |

Speculative decode varies strongly with text repetitiveness: the two rows
above were measured on real-text prompts; highly repetitive content (code,
template text) hits the chain drafter well and exceeds 60 tok/s in the same
configuration. Numbers vary with the weight format (GGUF / hgn); all dates
are in 2026.

## Features

- **Speculative decoding chain**: ngram drafts first, MTP as fallback, with
  round-by-round fallback; greedy does bit-exact comparison, sampling
  resamples and compares against the target distribution, so the output
  distribution is identical to serial decoding. Enabled by default, no
  parameters needed; the draft length γ is chosen per mode (fixed 4 for
  greedy, adaptive by acceptance rate for sampling) and can be pinned with
  `MTP_GAMMA=1-8`. Highly repetitive content (code comments, template
  text) shows significant measured speedups.
- **Standalone 8-bit MTP draft weights** sidecar, with a higher acceptance
  rate than the built-in 4-bit draft head.
- **Vision**: supports image input (OpenAI `image_url`), with KV reuse
  across turns.
- **OpenAI-compatible API**: streaming, tool calling,
  `/v1/chat/completions`.
- **256K context**, with a paged KV pool and two-tier prompt caching:
  in-RAM checkpoints at message boundaries (edit-and-resend replies
  instantly) plus KV snapshots recovered across restarts.
- **Concurrent requests**: multiple requests share one paged KV pool (4
  slots by default, 256K total — similar to llama.cpp's shared context);
  the GPU round-robins between requests and each request's output is
  bit-identical to running alone; long-prompt prefill is chunked to yield
  the GPU, capping other sessions' worst stall at ~0.6 s. Configuration
  and semantics in [CONCURRENCY.md](CONCURRENCY.md) (Chinese).
- **Model conversion tool**: HF safetensors → `.hgn`; quantization needs no
  calibration data, and can be distributed for your own fine-tuned models
  (see CONVERT_EN.md).

## Requirements

- Linux + ROCm (HIP 7.x), GPU architecture `gfx1151`; or Windows + AMD GPU
  driver (the GPU needs VRAM carved out in BIOS), see the "Windows" section
- Available memory ≥ 100 GiB (68 GiB weights pinned (page-locked) + KV)
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
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models
```

See [CONVERT_EN.md](CONVERT_EN.md) for details.

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
without a console window: it only puts a tray icon in the notification area
(right-click: open dashboard / copy API URL / view logs / quit; double-click:
open dashboard), and output goes to `logs\`. For troubleshooting,
`start_win.exe --console` restores the console mode (Ctrl+C or closing the
window stops it). Configuration is **shared with Linux via
`service.conf`** (edit it to change the model file name or context
window); environment variables can temporarily override it. Clients
connect to `http://<host>:8731/v1`.

Distribution: copy `build/` + `start_win.exe` + `models/` to any gfx1151
Windows machine and it just works — **no ROCm/TheRock installation
needed**; only the AMD GPU driver, plus enough VRAM carved out for the GPU
in BIOS (a 256K context needs 96 GiB).

Differences from the Linux version:

- Only hgn weights are supported: usable VRAM on Windows is capped at
  about 96 GiB, and GGUF weights are larger (hgn saves ~11 GiB over GGUF)
  and do not fit — `start_gguf.sh` does not apply; hgn weights are
  produced by the conversion tool, see [CONVERT_EN.md](CONVERT_EN.md)
- Image decoding supports PNG/JPEG via stb_image (WebP not wired up)
- Prefill chunk defaults to 8192
- Cold loading reads the full weights from disk (minute-scale, progress
  shown in console/logs)
- The launchers do not enable `GDEC_GEMM_WMMA` or `GDEC_GDN_FUSED` (the
  self-written WMMA GEMM and fused GDN kernel already promoted on Linux
  launchers, worth ~8-10% PP combined but unverified under TheRock — so
  Windows prefill uses hipBLASLt plus the legacy GDN path)

Known issues (root causes unknown; there are quite a few quirks, fixes
pending, priority very low):

- VRAM allocations above 41 GiB or 63 GiB fail
- The model hangs during decode (suspected console output backpressure: once
  the console is paused by a click/selection, child processes block on logging.
  Fixed: the launcher is now a tray app whose logs bypass the console, and
  kvsnap no longer prints while holding its lock; pending verification)

Build details are in [BUILD_EN.md](BUILD_EN.md).

## Documentation

- [QUICKSTART_EN.md](QUICKSTART_EN.md) — build, launch, configuration
- [BUILD_EN.md](BUILD_EN.md) — build environment details and
  troubleshooting
- [CONVERT_EN.md](CONVERT_EN.md) — model conversion tool
- [MTP_EN.md](MTP_EN.md) — speculative decoding parameters and comparison
  methods
- [NGRAM_EN.md](NGRAM_EN.md) — ngram verification design, benefits, and
  known divergences
- [CONCURRENCY.md](CONCURRENCY.md) — concurrent requests (PARALLEL)
  configuration and semantics (Chinese)
- [HGN-FORMAT_EN.md](HGN-FORMAT_EN.md) — the `.hgn` weight container
  format
- [GGUF.md](GGUF.md) — running directly from llama.cpp GGUF weights
  (Chinese)
- [data/README_EN.md](data/README_EN.md) — numerical regression benchmark
  (data/qsa-oracle) description
- [PORTING-WINDOWS_EN.md](PORTING-WINDOWS_EN.md) — Windows porting notes
  and measurements

## Tests

```bash
bash build.sh test   # Kernel unit tests, no model loading, expect ALL PASS
```

## Acknowledgements

This project's implementation borrows from [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) by peonist-ai. The `.hgn` weight container format is halogen's checkpoint container format — see [HGN-FORMAT_EN.md](HGN-FORMAT_EN.md). Many thanks to the halogen authors.

GGUF support heavily references [gufo](https://github.com/gufo-org/gufo) (MIT license): the routed-expert F16 WMMA GEMM kernel is ported from its RoutedF16GEMMKernel (`src/gpu/parts/26_kernels_moe_gguf.inc`); the LUT-decoded variants for hgn q4cp / GGUF IQ4 follow the same pipeline (`27_kernels_moe_lut.inc`); the GGUF↔engine tensor transform semantics reference its reference.cpp (`src/gguf_map.h`). Prefill optimizations such as HC gate fusion and producer epilogues writing the next GEMM's input directly also borrow from gufo's approach (comparison analysis in [GUFO-GAP.md](GUFO-GAP.md), Chinese).

The ngram speculative drafting approach and the two-tier prompt cache borrow ideas from the open-source [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT license); GGUF weights and the vision tower (mmproj) reuse the very same files as llama.cpp. See the "Attribution" section of [NGRAM_EN.md](NGRAM_EN.md).

## License

[AGPL-3.0](LICENSE)
