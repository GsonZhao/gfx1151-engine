# gfx1151-engine

*English: [README_EN.md](README_EN.md)*

在单张 AMD Strix Halo APU(gfx1151)上运行 177B MoE 模型的本地推理引擎,
目标模型为 Qwen3.8-Flash-Next(qwen4_exp 架构)及其同结构微调。

路由专家以 4-bit 量化存放在主机内存,GPU kernel 直读,不需要大显存;常驻内存的权重
约 67–77 GiB(视权重格式,PLE n-gram 表留在磁盘按需读),整机 122 GiB 内存即可提供
256K 上下文。

## 性能实测

开发机器为 **GMK EVO-X2(AMD Ryzen AI Max+ 395,Strix Halo / gfx1151)**,
122 GiB 内存:

| 指标 | 数值 |
| --- | --- |
| Prefill(128K 上下文) | 约 1400–1470 tok/s |
| Decode(投机) | 8K/32K 上下文约 54–55 tok/s,64K 约 36–45 tok/s(视权重格式) |
| Decode(不投机) | 约 25–30 tok/s(视权重格式) |
| 平均功耗 | 约 120 W |
| 瞬时最大功耗 | 约 130 W(爆发持续几秒后回落至 120 W 左右) |

## 权重格式与质量

引擎支持两种权重格式,服务、API、投机解码完全相同,换启动器即可切换:

| | hgn 标准 | hgn 高质量(HQ) | GGUF UD-Q4_K_XL |
| --- | --- | --- | --- |
| 文件 | 当前默认(`qwen38-flash-next-w4b.hgn` + overlay) | 从原始权重转换,见 [HGN-HQ.md](HGN-HQ.md) | Unsloth 发布,与 llama.cpp 同一份文件 |
| 路由专家 | 4-bit(q4cp) | 4-bit(q4cp,imatrix 加权) | Q4_K / Q5_1 为主 |
| dense(注意力、GDN、shared expert、embed、lm_head) | 4-bit | 8-bit(q8g32 overlay) | 8-bit(Q8_0) |
| KLD vs BF16(越低越好) | 0.163 | **0.0558** | 0.0511 |
| top1 与 BF16 一致 | 86.8% | 92.4% | 92.6% |
| 常驻权重 bpw / 大小 | 4.55 / 66.6 GiB | 4.70 / 68.8 GiB | 5.25 / 76.9 GiB |
| Prefill(8K prompt,chunk 2048) | 约 1200 tok/s | 约 1200 tok/s | 约 1200 tok/s |
| Decode(不投机) | 约 30 tok/s | 约 25 tok/s | 约 25 tok/s |
| 启动 | `start_hgn.sh` | `start_hgn.sh`(改 `MODEL_FILE` / `OVERLAY_FILE`) | `start_gguf.sh` |
| Windows | 支持 | 支持(尚未实测) | 不支持 |

- KLD:BF16 基准,wikitext-2 64 chunk × 512,测法与 unsloth / llama.cpp 相同(见 [KLD.md](KLD.md));
  llama.cpp 跑同一份 GGUF 为 0.049。
- 质量差距几乎全部来自 dense 的位宽:dense 改成 8-bit 后 KLD 0.163 → 0.063,imatrix 专家再降到
  0.0558。代价是 decode 每 token 读量增加,慢约 16%(与 GGUF 相同);prefill 不受影响。
- 常驻权重不含 PLE n-gram 表(hgn fp8 47.7 GiB,GGUF IQ4_NL 26.8 GiB),该表留在磁盘按需读。
  `python3 tools/bpw.py` 按类别复核各文件的 bpw(只读文件头,几秒)。
- 默认配置仍是 hgn 标准。HQ 文件已通过 `tools/hq_verify.sh`;部署方法见
  [HGN-HQ.md](HGN-HQ.md) 第 6 节。

## 特性

- **投机解码 chain**:ngram 优先起草、MTP 兜底,逐轮回退;贪心逐位比对,
  采样按目标分布重采样比对,输出分布与串行一致。默认生效,无需参数。
  高度重复内容(代码注释、模板文本)实测显著加速。
- **独立 8-bit MTP 草稿权重** sidecar,接受率高于内置 4-bit 草稿头。
- **视觉**:支持图像输入(OpenAI `image_url`),KV 复用跨轮生效。
- **OpenAI 兼容 API**:流式、工具调用、`/v1/chat/completions`。
- **256K 上下文**,两级 prompt 缓存:消息边界的内存检查点(编辑重发秒回)
  + KV 快照跨重启恢复。
- **两种权重格式**:自有 `.hgn`(Linux / Windows)与 llama.cpp 的 GGUF
  (Unsloth UD-Q4_K_XL,Linux),对比见上。
- **模型转换工具**:HF safetensors → `.hgn`,默认输出高质量版(8-bit dense overlay
  + 加权 4-bit 专家);有 llama.cpp 格式的 imatrix 就用,没有也能转。可用于自己的
  同架构微调模型(见 [HGN-HQ.md](HGN-HQ.md)、[CONVERT.md](CONVERT.md))。

## 要求

- Linux + ROCm(HIP 7.x),GPU 架构 `gfx1151`;或 Windows + AMD 显卡驱动
  (GPU 需 BIOS 划分显存),见「Windows」一节
- 可用内存 ≥ 100 GiB(权重锁页:hgn 约 68 GiB、GGUF 约 80 GiB,另加 KV)
- 编译依赖:rocBLAS、hipBLASLt、rocPRIM;API 前端另需
  libpng、libjpeg、libwebp（nlohmann/json 已随仓库提供）

## 快速开始

```bash
bash build.sh        # 编译引擎 + API,输出在 build/
bash start_hgn.sh    # hgn 权重:加载模型并启动服务(读取 service.conf)
bash start_gguf.sh   # 或 GGUF 权重(Unsloth UD-Q4_K_XL,与 llama.cpp 同一份文件)
```

两个启动器都从 `./models` 读取权重,缺文件时列出缺失项并退出;配置集中在
`service.conf`(hgn、GGUF 各一段)。GGUF 见 [GGUF.md](GGUF.md)。
详见 [QUICKSTART.md](QUICKSTART.md)。

自有微调模型(HF safetensors,同架构)转换:

```bash
# 没有 imatrix
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models
# 有 imatrix(llama.cpp 格式,GGUF 或旧版 imatrix.dat)
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models --imatrix /path/to/imatrix.gguf
```

输出基座 `.hgn`、8-bit dense overlay、8-bit MTP 草稿、视觉塔、分词器,以及可直接运行的
`start.sh`;32 核约 1.5 小时,磁盘约 125 GiB,只依赖 numpy。`--classic` 为旧的无数据转换器
(输出与以前逐字节相同)。高质量转换见 [HGN-HQ.md](HGN-HQ.md),格式与旧转换器见
[CONVERT.md](CONVERT.md)。

## Windows

Windows 版与 Linux 版功能一致(引擎 + OpenAI API + 多模态),移植记录与
实测见 [PORTING-WINDOWS.md](PORTING-WINDOWS.md)。编译在 Git Bash 中执行
(或双击 `build_win.bat`,仅编译期需要 Git):

```bash
bash build_win.sh           # 引擎
bash build_win.sh api       # OpenAI API 前端
bash build_win.sh launcher  # 免脚本启动器 start_win.exe
```

日常运行双击 `start_win.exe`(原生 Win32,不需要 Git/PowerShell):拉起引擎
+ API 双进程,输出实时显示并写入 `logs\`,Ctrl+C 或关窗停止。配置与 Linux
**共用 `service.conf`**(换模型文件名、改上下文窗口都编辑它),环境变量可
临时覆盖。客户端连 `http://<主机>:8731/v1`。

分发:`build/` + `start_win.exe` + `models/` 拷到任意 gfx1151 Windows
机器即用,**无需安装 ROCm/TheRock**;仅需 AMD 显卡驱动,并在 BIOS 为 GPU
划分足够显存(256K 上下文需 96 GiB)。

与 Linux 版的差异:

- 只用 hgn 权重(GGUF 未移植)。高质量 hgn 换文件即可用:weight arena +2.2 GiB,
  256K / chunk 8192 估算约 93.2 GiB(上限 95),尚未在 Windows 实测
- 图片解码经 stb_image 支持 PNG/JPEG(WebP 未接)
- prefill chunk 默认 8192
- 冷加载为整权重读盘(分钟级,进度见控制台/日志)
- 启动器未开 `GDEC_GEMM_WMMA` 与 `GDEC_GDN_FUSED`(Linux 启动器已转正的
  自写 WMMA GEMM 与 GDN 融合 kernel,合计约 8-10% PP,TheRock 下未验证——
  故 Windows 端 prefill 走 hipBLASLt + 旧 GDN 路径)

编译细节见 [BUILD.md](BUILD.md)。

## 文档

- [QUICKSTART.md](QUICKSTART.md) — 编译、启动、配置
- [BUILD.md](BUILD.md) — 编译环境细节与排错
- [GGUF.md](GGUF.md) — GGUF 权重加载、与 hgn 的性能对比
- [HGN-HQ.md](HGN-HQ.md) — 高质量 hgn:一键转换(可选 imatrix)、结果与部署
- [CONVERT.md](CONVERT.md) — 模型转换工具
- [KLD.md](KLD.md) — 质量测试(KLD,与 unsloth / llama.cpp 同口径)
- [MTP.md](MTP.md) — 投机解码参数与对比方法
- [NGRAM.md](NGRAM.md) — ngram 验证的设计、收益与已知分歧
- [HGN-FORMAT.md](HGN-FORMAT.md) — `.hgn` 权重容器格式
- [data/README.md](data/README.md) — 数值回归基准(data/qsa-oracle)说明
- [PORTING-WINDOWS.md](PORTING-WINDOWS.md) — Windows 移植记录与实测

## 测试

```bash
bash build.sh test     # kernel 单测,不加载模型,预期 ALL PASS
python3 tools/bpw.py   # 统计 models/ 下各权重的 bpw(按类别,只读文件头)
```

质量(KLD)测试需要 BF16 基准,流程见 [KLD.md](KLD.md)。

## 致谢

本项目的实现方式借鉴了 peonist-ai 的 [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server);`.hgn` 权重容器格式即 halogen 的 checkpoint 容器格式(见 [HGN-FORMAT.md](HGN-FORMAT.md))。感谢 halogen 作者的工作。

ngram 投机解码的起草思路另借鉴了开源项目 [llama.cpp](https://github.com/ggml-org/llama.cpp)(MIT 许可证),声明详见 [NGRAM.md](NGRAM.md) 的「来源与声明」一节。GGUF 路由专家 WMMA kernel 移植自 gufo(MIT 许可证),见 [GGUF.md](GGUF.md)。

## 许可证

[AGPL-3.0](LICENSE)
