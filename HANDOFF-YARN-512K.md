# YaRN 512K 实机测试交接

目标：在 gfx1151 上验证分支 `codex/yarn-512k` 将原生 262144 上下文扩展到
524288，并确认默认短文本路径不回归。

## 2026-09-30 测试后状态

实机报告见 `YARN-512K-RESULTS.md`：Linux 单序列、BF16 分页 + WMMA + BTV
已通过编译、34K～500K 阶梯、三档 needle、短文本回归和 KVSNAP 恢复/隔离。
补测已覆盖完整 524288 边界、预算为 1/25 的精确裁剪，以及 350K 深处 PPL；
factor 1 的短文本 KLD 与基线近似一致，不应据此声称完整 logits 逐 bit 一致。

256K 离线 prefill 约为原生 1374.4 tok/s、YaRN 1370.3 tok/s，开销约 0.3%，
属于噪声级；两次生成 token 一致。

并发门禁旧版数据见报告 §13～§14，仅作历史记录。新版修复的复测见 §15：
`PARALLEL=2`、关闭快照，两条约 24K prompt 输出文本一致且 needle 各 3/3；
两条约 311K prompt 超订时，后发 **5.3s 在下一调度边界被拒**，先发 242.1s
完成且不被中断。五组复测均 PASS，不表示两条完整 512K 序列可同时驻留。

`MAX_CONTEXT=524288 KV_POOL_TOKENS=0` 时所有槽位共享一条 512K 大小的池。
`GDEC_KV_RESERVE_DECODE` 默认只预约 4096 token 的 decode 增长，不是输出上限；
更长生成仍可能触发中途淘汰/中断。RoPE 配置也是实例级，不支持逐请求切换。

## KV 准入修复：Linux 实机复测通过

2026-09-30 已修复代码并完成实机复测，结果见报告 §15；§14 属于修复前版本。
按报告和测试覆盖，本轮可验收 gfx1151/Linux 的 factor 2、512K、BF16 分页 +
WMMA + BTV 配置。Windows、YaRN M-RoPE 和其他未测组合不在本次验收范围。

- **同步取数**：准入先放开 `g_sm`，取得 GPU 所有权 `g_turn` 后再拿 `g_sm`
  并复核槽位/取消状态。不再读空闲 deque；成功准入直接把 GPU turn 交给生成，
  避免二次 acquire。拒绝和取消均释放 turn，不持有 `g_sm` 等 GPU。
- **重置记账**：删除旧 `pages0`，预约改为整条序列的目标页数。其他活跃槽占用按
  `max(目标页数, 当前 mapped 页数)` 扣除，旧槽 reset 释放的页不再被重复计入。
- **上下文边界**：预约按 MAX_CONTEXT 裁剪，移除固定 `+2` 页余量；并行模式下
  只有一条活跃请求时，也能把整池用满。
- **保守策略**：活跃序列分别记账，不依赖前缀共享省出的物理页；空闲槽和检查点
  独占页仍由既有压力处理逐出。decode 超预约上限仍可能触发中途淘汰/中断。

新增 `tools/kv_admission_test.py` 直接抽取生产准入代码，用假模型运行主机测试；
Windows LLVM Clang 22.1.8 下七组全 PASS，包括旧 400 页 → 新请求当前 80 页 /
目标 90 页的误拒回归、GPU 同步、拒绝/取消释放、边界，以及四线程 400 次 churn。
`build.sh test` / `build_win.sh test` 已接入。报告 §15 已确认远端 g++ 主机回归、
主机 TSAN、HIP 重编与 ktest 全 PASS。TSAN 仅覆盖假模型下的生产准入代码，
不是整个 HIP 引擎的无竞态证明。快速拒绝要等 GPU 安全调度边界，不保证 0.3s。

### 已完成复测的重跑清单

以下五组已按报告 §15 通过，保留用于后续代码改动的回归，不是待完成事项。
COW 与 SSD 恢复的生成 token 对拍按报告完成；新增辅助脚本部分只打印结果，
单凭脚本退出码不能替代日志里的槽位复用、缓存层来源和输出一致性检查。

先停止生产服务，确认检出的测试版本包含 `src/kv_admission.h` 和两份主机测试文件。
报告 §15 保留了测试当时基于 `78a41cc` 的未提交工作区状态；后续重跑应使用
包含本轮修复的合并提交，不要只检出报告里的旧基线 commit。

```bash
python3 tools/kv_admission_test.py --cxx g++
GPU_ARCH=gfx1151 bash build.sh all
GPU_ARCH=gfx1151 bash build.sh test
# 可选：Linux 编译器支持时检查主机数据竞态
python3 tools/kv_admission_test.py --cxx g++ --tsan
```

首次起服关闭 RAM/SSD 快照；保持 BF16 KV + WMMA + BTV，不强制开启 UNION：

```bash
MAX_CONTEXT=524288 ROPE_FACTOR=2 ROPE_ORIGINAL_CTX=262144 \
ROPE_BETA_FAST=32 ROPE_BETA_SLOW=1 PARALLEL=2 KV_POOL_TOKENS=0 \
RCKPT_MAX=0 KVSNAP_MAX_GB=0 GDEC_KV_RESERVE_DECODE=4096 \
bash start_hgn.sh --check
```

检查后用同样环境去掉 `--check` 起服，再执行：

```bash
python3 tools/yarn_conc.py consistency
python3 tools/yarn_conc.py contend
python3 tools/yarn_boundary.py oversize,exact-full,budget-1,budget-25
```

重点判定（每组结束后再发小请求并检查健康，不接受只看 HTTP 错误码）：

1. **一致性和超订**：A/B 串行与并发输出一致、needle 正确；超订的后来者在下一
   调度边界准入拒绝，不做长 prefill；先发不被中断，服务之后仍可用。
2. **两个非空槽的非前缀复用**：先并发填满两个槽的部分历史，再用不同 prompt
   复用旧槽。构造第一条从旧 400 页重置到目标 90 页、第二条目标 1800 页；合计
   1890 < 2048 应准入。用引擎 token-ID 协议更易精确控制，并记录 slot/reset 日志，
   确认实际复用了两个非空槽，而不是只命中空槽或 live prefix。
3. **并行满池边界**：PARALLEL=2 但没有其他活跃请求，budget-1/25 应成功生成
   精确预算；exact-full/oversize 应干净拒绝。旧边界只测了 PARALLEL=1。
4. **RAM 检查点和 SSD 恢复**：重启启用 RCKPT_MAX=8、KVSNAP_MAX_GB=20，重复
   一致性/槽位复用；增加前缀共享、partial-page COW、池压力逐出和恢复后再准入。
   不应死锁、非法访问或旧页误计；容量按活跃序列保守记账，共享前缀并不保证可超订。
5. **超预约 decode 和取消**：停止 API/引擎后跑下方专用套件，检查运行中/排队
   取消、断连、重复 GEN、多路 churn 和硬兜底中断；受害者退出后池必须可继续复用。

```bash
NEW_BIN=build/gdec bash tools/conc_verify.sh
```

该套件会自行起测试引擎。溢出阶段已固定 `GDEC_KV_RESERVE_DECODE=256`，让 O1
先准入再超过预约、覆盖中途 starve；O2 继续覆盖准入拒绝。不要用默认 4096 覆盖
这阶段，否则可能在准入时就拒绝 O1，无法验证硬兜底。

Windows 和 YaRN M-RoPE 可另行验证；本轮无需重跑完整长上下文质量/PPL 阶梯。

### 验收范围与非阻塞项

- 此轮没有报告新增 KV 准入阻塞项；页池仍是多槽共享，decode 超预约上限仍可能
  中断后来者，不能把准入 PASS 当作无限生成或两条完整 512K 同驻留的保证。
- 合成 prompt 的负 draft acceptance 属统计异常。生成对拍已通过，不阻塞此次
  KV/YaRN 验收；其具体统计根因未由本轮测试证明，留作独立排查。
- `/cache` CSTAT 超时仍属既有协议问题，不在本分支修复范围。

## 其他既有问题

`/cache` 的 CSTAT 超时为既有协议问题，本分支不顺带修复。

以下保留原始测试清单，不表示所有项目都还需要重跑。

## 当前实现

- 推荐配置：`ROPE_FACTOR=2`、`ROPE_ORIGINAL_CTX=262144`、`ROPE_BETA_FAST=32`、`ROPE_BETA_SLOW=1`。
- 默认 `ROPE_FACTOR=1`，表示原生 RoPE；短文本对比测试必须使用该默认值作为基线。
- 512K 会自动关闭 `GDEC_QSA_UNION`，不要手动强制打开。
- 512K 推荐 BF16 KV：`GDEC_QSA_KV_BF16=1`、`GDEC_QSA_WMMA=1`、`GDEC_QSA_WMMA_BTV=1`。
- YaRN attention factor 只应用于主 attention Q/K；indexer partial RoPE 不应用该缩放。

## 测试顺序

先停止已有 engine/API，再执行：

```bash
GPU_ARCH=gfx1151 bash build.sh test
GPU_ARCH=gfx1151 bash build.sh all
```

检查启动配置，不要直接启动：

```bash
unset GDEC_QSA_UNION
MAX_CONTEXT=524288 ROPE_FACTOR=2 ROPE_ORIGINAL_CTX=262144 \
ROPE_BETA_FAST=32 ROPE_BETA_SLOW=1 PARALLEL=1 KV_POOL_TOKENS=0 \
bash start_hgn.sh --check
```

检查通过后，用完全相同的环境变量启动实际服务。首次测试建议单并发、单序列，
避免把分页 KV、并发调度和 YaRN 问题混在一起。

## 必测项目

1. **启动与内存**：确认日志打印 YaRN factor、KV 类型、分页状态；观察是否出现
   arena 不足、hipMalloc 回退、OOM、非法访问或长时间 stall。
2. **长度阶梯**：依次测试约 32K、128K、256K、300K、400K、512K prompt；每档
   至少生成少量 token，确认 prefill 完成且 decode 能继续。
3. **长上下文效果**：在 300K 以上做 needle 定位；同时记录 PPL/KLD（若已有对应
   工具和基准），重点看 256K 之后是否出现明显退化或错位。
4. **短文本回归**：factor 1 与 factor 2 分别跑相同短 prompt，比较输出、PPL/KLD
   和延迟；factor 1 必须保持原生基线行为。
5. **KV 快照**：YaRN 配置下写入 KVSNAP，重启后恢复并命中；再切换 factor 1，
   确认不会命中 YaRN 产生的旧 KV 页。
6. **分页 BF16**：确认 512K BF16 分页能完成写入、读取、decode 和 reset；不要
   同时开启 `GDEC_QSA_UNION`、`GDEC_QSA_WMMA6` 等未验证组合。

## 记录与判定

每次测试请保存：完整 engine/API 日志、启动环境变量、git commit/工作区状态、
prompt token 数、prefill/decode 耗时、峰值显存/RAM、是否发生 arena 回退，以及
最终输出或 PPL/KLD 结果。

判定为通过的最低条件：512K prefill 和 decode 无错误；300K+ needle 能定位；
KVSNAP 隔离正确；factor 1 短文本基线无回归；512K BF16 内存未发生不可接受的
回退或 OOM。任何 `hipErrorIllegalAddress`、NaN/Inf、位置错位、服务无响应或
旧 KV 错命中都应立即停止该组合并保留日志。

## 回传格式

请回传以下信息，便于继续定位：

```text
GPU / ROCm:
git revision:
config:
prompt tokens:
KV mode / paging:
prefill time:
decode result:
memory / arena fallback:
KVSNAP result:
errors or log excerpt:
```
