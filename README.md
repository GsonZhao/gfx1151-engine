# gfx1151-engine

*English: [README_EN.md](README_EN.md)*

在单张 AMD Strix Halo APU(gfx1151)上运行 177B MoE 模型的本地推理引擎,
目标模型为 Qwen3.8-Flash-Next(qwen4_exp 架构)及其同结构微调。

68 GiB 量化权重以 4-bit 量化存放在主机内存,GPU kernel 直读,不需要大显存;
整机 122 GiB 内存即可提供 256K 上下文。

## 性能实测

开发机器为 **GMK EVO-X2(AMD Ryzen AI Max+ 395,Strix Halo / gfx1151)**,
122 GiB 内存:

| 指标 | 数值 |
| --- | --- |
| Prefill(128K 上下文) | 约 1400–1470 tok/s(09-28 实测) |
| Decode(投机,greedy,γ=4) | 8K 约 45、64K 约 46 tok/s(09-29 真实文本) |
| Decode(投机,采样,自适应 γ) | 8K 约 40、64K 约 45 tok/s(09-29 真实文本) |
| Decode(不投机) | 约 25–30 tok/s(09-28 实测,视权重格式) |
| 平均功耗 | 约 120 W |
| 瞬时最大功耗 | 约 130 W(爆发持续几秒后回落至 120 W 左右) |

投机 decode 随文本重复度变化很大:上面两行用真实文本 prompt 测得;高重复内容(代码、
模板文本)chain 起草命中率高,同配置实测可达 60 tok/s 以上。权重格式(GGUF / hgn)不同
数字会有出入;表中 09-28/09-29 均为 2026 年。

## 特性

- **投机解码 chain**:ngram 优先起草、MTP 兜底,逐轮回退;贪心逐位比对,
  采样按目标分布重采样比对,输出分布与串行一致。默认生效,无需参数;
  草稿长度 γ 按模式自选(greedy 固定 4、采样按接受率自适应),也可用
  `MTP_GAMMA=1-8` 固定。高度重复内容(代码注释、模板文本)实测显著加速。
- **独立 8-bit MTP 草稿权重** sidecar,接受率高于内置 4-bit 草稿头。
- **视觉**:支持图像输入(OpenAI `image_url`),KV 复用跨轮生效。
- **OpenAI 兼容 API**:流式、工具调用、`/v1/chat/completions`。
- **256K 上下文**,分页 KV 页池 + 两级 prompt 缓存:消息边界的内存检查点
  (编辑重发秒回) + KV 快照跨重启恢复。
- **并发请求**:多条请求共享同一个分页 KV 页池(默认 4 路、总共 256K,
  类似 llama.cpp 的共享上下文),GPU 按请求轮转,每条输出与单独运行逐位
  一致;长 prompt prefill 分段让出 GPU,其它会话最长卡顿约 0.6 s。配置与
  语义见 [CONCURRENCY.md](CONCURRENCY.md)。
- **模型转换工具**:HF safetensors → `.hgn`,量化无需校准数据,
  可分发给自己的微调模型使用(见 [CONVERT.md](CONVERT.md))。

## 要求

- Linux + ROCm(HIP 7.x),GPU 架构 `gfx1151`;或 Windows + AMD 显卡驱动
  (GPU 需 BIOS 划分显存),见「Windows」一节
- 可用内存 ≥ 100 GiB(权重 68 GiB 锁页 + KV)
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
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models
```

详见 [CONVERT.md](CONVERT.md)。

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
+ API 双进程,不开控制台,只在任务栏右下角放托盘图标(右键:打开面板 / 复制
API 地址 / 查看日志 / 退出;双击:打开面板),输出写入 `logs\`;排查问题可用
`start_win.exe --console` 回到控制台模式(Ctrl+C 或关窗停止)。配置与 Linux
**共用 `service.conf`**(换模型文件名、改上下文窗口都编辑它),环境变量可
临时覆盖。客户端连 `http://<主机>:8731/v1`。

分发:`build/` + `start_win.exe` + `models/` 拷到任意 gfx1151 Windows
机器即用,**无需安装 ROCm/TheRock**;仅需 AMD 显卡驱动,并在 BIOS 为 GPU
划分足够显存(256K 上下文需 96 GiB)。

与 Linux 版的差异:

- 只支持 hgn 权重:Windows 下可用显存上限约 96 GiB,GGUF 权重体积更大
  (hgn 比 GGUF 省约 11 GiB)放不下,`start_gguf.sh` 不适用;hgn 权重由
  转换工具生成,见 [CONVERT.md](CONVERT.md)
- 图片解码经 stb_image 支持 PNG/JPEG(WebP 未接)
- prefill chunk 默认 8192
- 冷加载为整权重读盘(分钟级,进度见控制台/日志)
- 启动器未开 `GDEC_GEMM_WMMA` 与 `GDEC_GDN_FUSED`(Linux 启动器已转正的
  自写 WMMA GEMM 与 GDN 融合 kernel,合计约 8-10% PP,TheRock 下未验证——
  故 Windows 端 prefill 走 hipBLASLt + 旧 GDN 路径)

已知问题(原因均未查明;疑难杂症较多,待解决,优先级很低):

- 显存分配超过 41 GiB 或 63 GiB 时失败
- 模型 decode 过程中卡死(疑为控制台输出反压:控制台被点选暂停后,子进程写日志
  阻塞。已修复:启动器改为托盘程序、日志不经过控制台,kvsnap 不再持锁打印;待验证)

编译细节见 [BUILD.md](BUILD.md)。

## 文档

- [QUICKSTART.md](QUICKSTART.md) — 编译、启动、配置
- [BUILD.md](BUILD.md) — 编译环境细节与排错
- [CONVERT.md](CONVERT.md) — 模型转换工具
- [MTP.md](MTP.md) — 投机解码参数与对比方法
- [NGRAM.md](NGRAM.md) — ngram 验证的设计、收益与已知分歧
- [CONCURRENCY.md](CONCURRENCY.md) — 并发请求(PARALLEL)的配置与语义
- [HGN-FORMAT.md](HGN-FORMAT.md) — `.hgn` 权重容器格式
- [GGUF.md](GGUF.md) — 直接用 llama.cpp GGUF 权重运行
- [data/README.md](data/README.md) — 数值回归基准(data/qsa-oracle)说明
- [PORTING-WINDOWS.md](PORTING-WINDOWS.md) — Windows 移植记录与实测

## 测试

```bash
bash build.sh test   # kernel 单测,不加载模型,预期 ALL PASS
```

## 致谢

本项目的实现方式借鉴了 peonist-ai 的 [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server);`.hgn` 权重容器格式即 halogen 的 checkpoint 容器格式(见 [HGN-FORMAT.md](HGN-FORMAT.md))。感谢 halogen 作者的工作。

GGUF 支持大量参考了 [gufo](https://github.com/gufo-org/gufo)(MIT 许可证):路由专家 F16 WMMA GEMM kernel 移植自其 RoutedF16GEMMKernel(`src/gpu/parts/26_kernels_moe_gguf.inc`),hgn q4cp / GGUF IQ4 的 LUT 解码版沿用同一条流水线(`27_kernels_moe_lut.inc`),GGUF 与引擎张量间的变换语义参考其 reference.cpp(`src/gguf_map.h`);prefill 的 HC 门控融合、生产者 epilogue 直写下一 GEMM 输入等优化也借鉴了 gufo 的做法(对照分析见 [GUFO-GAP.md](GUFO-GAP.md))。

ngram 投机解码的起草思路与两级 prompt 缓存借鉴了开源项目 [llama.cpp](https://github.com/ggml-org/llama.cpp)(MIT 许可证),GGUF 权重与视觉塔(mmproj)直接复用 llama.cpp 的同一份文件;ngram 声明详见 [NGRAM.md](NGRAM.md) 的「来源与声明」一节。

## 许可证

[AGPL-3.0](LICENSE)
