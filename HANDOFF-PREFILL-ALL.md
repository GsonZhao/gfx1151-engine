# prefill 全方位优化方向（2026-10-03，交接给 Windows 实机）

范围：所有权重与路径——w4b hgn（q4cp + overlay）、hgn v2（ht/i4r/q6g64）、GGUF（仅 Linux）；
短 prompt / 多轮增量 prefill、16K chunk 稳态、32K–256K 长上下文，以及"少做 prefill"（缓存命中）。

本文汇总了下面几份文档里还没做完的部分，并补上了它们都没覆盖的短 prompt 场景：
HANDOFF-V2-PREFILL.md（10-02/03 Windows 实测）、HANDOFF-PREFILL-NEXT.md（09-28）、
PREFILL.md §10–11、GUFO-GAP.md（09-25）、KV 前缀复用遗留（09-26）。**已否决项统一列在 §8，
动手前先查一遍。**

---

## 0. 现状速览（数字来自不同日期/构建，只用来看差距的量级）

| 场景 | 我们 | 参照 | 差距 |
|---|---|---|---|
| 128K，chunk 16384 | v2 Win 1439 / w4b Linux 1446 / GGUF Linux 1451 | halogen 1517 | ~5% |
| 32K，chunk 16384 | v2 Win ~1521 / w4b Linux 1512 | halogen 1567 | ~3% |
| 8K prompt，chunk 2048 | w4b Linux ~1195（fdf805a，09-25） | gufo d0 1628 | **~36%** |
| 32K prompt，chunk 2048 | GGUF Linux 1152（09-25） | gufo d32K 1422 | **~20%+** |
| 64K | w4b/v2 ~1495（chunk 16384） | gufo d64K 1304（chunk 2048） | 我们领先 |
| 小 P TTFT（M1b 口径，P=2048 单 chunk） | v2 Win **1212**（F1 后；F1 前 1066） | gufo pp2048@d0 1628 | **25.6%**（F1 前 34.5%） |

结论：大 chunk / 长上下文只差 3–5%，剩下的都是 1–3% 一项的细活。**真正的大缺口在小 P**：
gufo 在 2048 chunk 就能跑到 1400–1600，我们要靠 16K chunk（多占 ~7.5 GiB workspace）才追平。
用户实际碰到的大多是小 P：一条 3K 的 prompt 只有一个 P=3000 的 chunk；多轮对话每轮只新增
几百到几千 token。

小 P 这块**从没做过修好 MoE 之后的 profile**。GUFO-GAP.md 的逐组件对照是 09-25 修 MoE 之前做的
（当时 MoE 用标量 kernel，占差距的 78%）。按那张表扣掉 LUT kernel 的收益，2048 chunk 下剩余的
~0.22 ms/token 差距大致来自：dense +0.06、注意力 +0.06、MoE +0.04，另有约 0.06 说不清。
这只是推算，必须先测（M1）。

---

## 1. 总排名

"预期"栏如无特别说明，指 16K chunk 稳态整体 pp 的变化；小 P 的收益另行标注。

| # | 方向 | 适用 | 预期 | 工作量 | 风险 | 前置 |
|---|---|---|---|---|---|---|
| M1 | **小 P 扫描 + KPROF 分段**（P=256…4096，base 0/32K） | 全部 | 无直接收益，决定 A 轨排序 | 半天 | 无 | 代码回同步 |
| M2 | w4b 在 Windows 跑一遍 KPROF（16384 / 8192） | w4b | 同上 | 1 小时 | 无 | — |
| M3 | KLD 基础设施：bf16_c512.kld 拷到 Windows；补一份长上下文参考 | 全部 | 解锁 B1/B3/C1 | 拷文件；长参考需 BF16 模型 | 无 | — |
| A1 | **q4cp 码本改寄存器 perm 查表**（i4r ③ 的 q4cp 版） | w4b（日后 GGUF IQ4） | 2–3%，小 P 更多 | 小–中 | 低，逐 bit | — |
| D1 | **decode 吐出 `<tool_call>`/`</think>` 时存 rckpt** | 全部（agent） | 每个 agent 回合省一次"上条回复"的重 prefill | 中 | 中 | — |
| A2 | 小 P 的 MoE：tile 粒度 / 解码摊薄 | 全部 | 小 P 待 M1；16K ≈0 | 中 | 低 | M1 |
| A3 | 小 P 的 dense：q4cp/q8g32 直读量化 GEMM，免 `d_wbf16` 往返 | w4b、GGUF（v2 ht 难） | P≤2048 时 2–12%；16K <1% | 中 | 低 | M1 |
| B1 | down pairs 改 f16：v2 默认开的决定 + w4b 同款 | v2、w4b | 各 ~1.3% | 小（v2 已实现） | KLD | M3 |
| B2 | HC 链：reduce 并进 gr_write、去掉 Rhat 落盘、up-fused GEMM 调优 | 全部 | 合计 1.5–3% | 中 | 部分非逐 bit | — |
| C1 | indexer 打分改 bf16 WMMA，score 存 bf16 | 全部 | 128K ~3.5%，32K ~0.8% | 中 | KLD（长上下文） | M3 长参考 |
| B3 | dense k1 形状 (2560,6144) split-K / stream-K | 全部 | ≤1.4% | 中 | 低 | — |
| C2 | gdn:scan 深流水 | 全部 | ≤3.5% | 数天 | 收益不确定 | — |
| B4 | kMid occupancy（VGPR 169，plane 写 16-way bank conflict） | v2 | ~1% | 中 | 低 | — |
| D2 | TokenCache 落盘（服务重启后长对话不再全量重算） | 全部 | 重启后首轮省整段 prefill | 中 | 低 | — |
| E1 | Windows 按 arena 估算自动选 chunk 16384；验 256K@16384 | Windows | +1.7%（拿内存换） | 小 | arena 余量 | — |
| B5 | rot_in 并进 norm | v2 | 0.5% | 小 | 低 | — |
| §10 | **下一步方向（10-03 记录，全部要做）**：D1 rckpt / 小 P gemv 上限 / C2 / 降级排查 / 小项 | — | 见 §10 | 见 §10 | — | — |

建议顺序：**同步代码 → M1+M2（同一天出数据）→ A1（与 M 系列并行，microbench 不依赖 M1）
→ 按 M1 数据在 A2/A3 中挑 → D1 → B 系列 → C 系列**。M3 的拷文件今天就做，否则 B1/C1
永远卡在门槛上。

---

## 2. 先做：测量

### 2.0 前置：Windows 代码回同步

Windows 上的改动**还不在本地仓库**（本地没有 `07_kprof.inc`，`27_kernels_moe_lut.inc` 里
也没有 kMid）。需要回同步并提交的有：

- KPROF：`07_kprof.inc` 与各处 `KP`/`KP_SUB` 埋点
- kMid：`moe_lut_up_mid`
- i4r 算术解码与 scale 缓存
- `GDEC_V2_PAIRS_F16`、`GDEC_PAD_BATCH`
- 工具：`tools/gdn_fused_proto.cu`、`tools/idx_s8.cu`、`tools/pad_sweep.sh`、`moe_lut_test --v2/--opt`、`pp_win.sh` 读 `PREFILL_CHUNK`

建议拆成 3–4 个提交：KPROF / MoE kernel / 开关 / 工具。下面所有方向都基于同步后的代码。

### M1：小 P 扫描（最重要）

目的：给出 P 从 256 到 16384、每 token 每分段的耗时（ms/token），回答小 P 下时间到底花在哪。

做法（一个前台脚本，例如 `tools/pp_psweep_win.sh`，结尾打印汇总表）：

1. 固定 maxctx（如 139264），对 P ∈ {256, 512, 1024, 2048, 4096, 16384}：
   - **base=0**：prompt 长度取 3×P，`PREFILL_CHUNK=P`，取第 2、3 个 chunk 的均值。
     第 1 个 chunk 有 PLE 冷读，不要用。
   - **base≈32K**：先 prefill 32K，再续一段长度为 P 的 prompt，模拟多轮增量。这一步最能
     反映真实 TTFT。离线 bench 如果不支持续写，就走 API 发两次请求：第二次带相同前缀，
     命中 live prefix 后只 prefill 新增的 P 个 token，`max_tokens=1`，看引擎日志里的
     prefill 时间。
2. 每档同时开 `GDEC_KPROF=1` 和 `GDEC_PROF=1`。后者记下 host 侧的 `ple_wait`、`gemm_host`、
   argmax_d2h 等待。
3. 汇总表的列：P、tok/s、各分段 ms/token（gdn/qsa/moe:up/down/reduce/shared/hc/ht_deq），
   以及 **segsum 与墙钟之差**（即 GPU 空转和 host 开销）。
4. v2 和 w4b 各跑一遍。

要特别看的几个数：

- **ht_deq / q4cp 反量化**：`gemm_wbf16`（40_model.inc ~1883）对每个 dense gemm 都整张反量化
  到 `d_wbf16`，开销按 chunk 计，与 P 无关。16K 时 v2 只占 0.7%，但 P=512 时同样的
  ~70 ms 可能占到 10%+。超过 5% 就做 A3。
- **moe:up+down 的等效带宽**：P≲1024 时几乎每个专家都被选中（P=512 → 每专家平均 10 行，
  一个也不落的概率 e^-10），所以每个 chunk 都要把 ~64 GB 专家权重整个读一遍，物理下限约
  0.28 s/chunk。实测远高于这个数就做 A2。
- **segsum 与墙钟之差**：超过 5% 才考虑 launch/host 方向（hipGraph 之类）；否则这条不做。
- **Windows 的 `ple_wait`**：Windows 主机内存只有 32 GB，51 GB 的 n-gram 表放不进页缓存，
  短 prompt 每个 chunk 都要从 SSD 取行。Linux 测的 0–6 ms 不能代表 Windows。如果这里有
  几十毫秒，那是一条新的独立方向（预取提前到 tokenize 后立即开始）。

**状态（10-03，已完成，v2 + w4b 双跑）**：全表与细分在 `logs/m1_m2_report.md`。四条关键
结论：① A3 触发——v2 ht_deq P=512 占 13.3%（P=256 19.5%），w4b 边缘（4.8%）；② A2 双
权重都不触发——小 P 的 moe:up+down 实测低于"全专家重读"朴素下限（router 偏斜），没有
低于物理下限的浪费；③ gap%≈0 全档成立（v2/w4b 同）——hipGraph 维持否决被测量坐实；
④ Windows ple_wait 远超预期（v2 P=512 达 661 ms ≈ 该档墙钟 95%）——PLE 预取提前到
tokenize 后立即开始是**新独立方向**，且会截顶 A1/A3 的小 P 收益。脚本
`tools/pp_psweep_win.sh`，日志 `logs/m1_p*.log`、`logs/m1_w4b_p*.log`。

**④ 复核（10-03 下午，三连复现）：661 ms 未复现，判定偶发，新方向撤回。** v2 同脚本
P=256/512/1024 连跑 3 遍（14:01–14:05，间隔 ~85 s，每遍进程级全新），首真实 chunk 的
一次性 PLE 停顿稳定为 **35 / 64–66 / 120 ms**（三遍一致），与 w4b 当日的 37–120 ms 逐档
吻合；后续 chunk ple_wait 不再增长（被 GPU 遮住）。661 的出现条件是**表文件页缓存冷**
（M1 的 v2 跑是当天 01:44 首个 v2 进程；复现前表已被当天 KLD/pp 多次触碰）——即"每天
首跑/重启后一次性 ~0.6 s"，不是稳定现象。注意 ple_wait 是**累计值**，661 与 68 都是首
chunk 一次性付出的。另核实：v2 与 w4b 的 PLE 表是**同一张**（dtype 10、
dims (128,2500012,160)、47.684 GiB，采样行逐字节一致）——v2 独立成
`qwen38-flash-next-ngram.hgn`（51.2 GB 文件几乎全是这张表），w4b 内嵌在 base .hgn 偏移
1.88 GiB 处；`w4b.overlay.hgn`（2.4 GiB）是权重 overlay，与 PLE 无关，两边 ple_wait
无模型间差异。结论：稳定态每进程只在首 chunk 付一次 ~P×0.125 ms（P=512 ≈ 单 chunk 墙钟
9%，会话级可忽略；16K 档完全被遮住），只有冷启动才有 ~0.6 s 级停顿——**"PLE 预取提前
到 tokenize 后"不值得做**，A1/A3 的小 P 收益不存在 PLE 墙截顶。日志
`logs/m1_plerep{1,2,3}_p*.log`，勘误同步在 `logs/m1_m2_report.md` ④。

### M2：w4b 在 Windows 的 KPROF

KPROF 目前只测过 v2。w4b 的 MoE 走 q4cp LUT 查表，不走算术解码，分段比例不同。A1 的预期
收益就取决于 w4b 的 moe:up+down 占比。16384 和 8192 各跑一次 128K 即可，也顺便核实
PORTING-WINDOWS.md 里"w4b Windows 慢 25%"在现代码上是否还成立（v2 已证明不成立）。

**状态（10-03，已完成）**：128k 稳态 @16384 = 1412.4 tok/s（n=7 剔前 2），@8192 = 1391.2。
**"w4b Windows 慢 25%"不成立**：vs Linux 1446 只差 −2.3%（v2 同口径 1439 vs 1444 也追平）。
分段：moe 33.7%（up 1791.7 + down 1097.5 + shared 350.5 + reduce 451.2 + route 63.4，
子段埋点已随 M1-w4b 进源码）、gdn 28.4%、qsa 22.9%。ple_wait 2272 ms/chunk 被 GPU 完全
遮住（segsum≈wall）。全表见 `logs/m1_m2_report.md`，日志 `logs/m2_w4b_128k_c{16k,8k}.log`。

### M3：KLD 基础设施

- **立刻做**：把 Linux 上的 `~/Workspace/gfx-1151-kvsnap/data/kld/bf16_c512.kld` 拷到 Windows。
  B1（f16 pairs）就卡在这里：自 A/B 的 0.0204 无法判断是否过线，需要 0.10699+0.002 这条绝对门槛。
- **长上下文参考**：C1 以及 B2 里非逐 bit 的部分，需要 ≥32K 的 KLD 参考。c512 的 64×512 测不到
  indexer 选块（2051 以内全选）。这份参考要用 BF16 模型在 Linux 上生成：ctx 32768，chunks 4–8，
  文件会很大，先按词表 top-k 稀疏存一份，确认 `48_kld.inc` 支持这种格式再生成。这是 C1 的
  硬前置；在它到位之前，C1 只能停在 microbench。

---

## 3. A 轨：短 prompt / 增量 prefill

### A1. q4cp 码本改寄存器 perm 查表（推荐最先动手的 kernel 项）

现状：`k_moe_lut` 的 q4cp 分支（27_kernels_moe_lut.inc ~248 行）每 32 个权重做 16 次
`s_cbp[byte]` LDS 查表，再乘一次 `__hmul2`。i4r 分支已在 Windows 改成 `0x6400+nib` 的算术解码，
microbench −10.9%，且逐 bit 一致。q4cp 的码本是非线性的，不能照抄算术解码。但它是**整张张量
共用的 16 项 fp32 码本**（文件头注释："per-tensor 16 x fp32 codebook"），在 block 内是常量。

做法：

- 16 项转成 f16 后拆成低字节表、高字节表，各 16 字节。放进 SGPR 即可（码本一致，不占 VGPR），
  不占 VGPR 就不会重蹈 ①② 的 occupancy 覆辙。
- `v_perm_b32` 一次能从 8 字节里按 4 个选择子取 4 个字节，所以 16 项的表需要两次 perm 加
  一次按 bit3 的 `v_cndmask`（或 `v_bfi`）。低、高字节各做一遍，再用一次 perm 交织成 2 个
  half2。估计每 4 个权重 6–8 条 VALU，现在是每 4 个权重 2 次 LDS 读加 2 次 hmul2。
- `__floats2half2_rn` 舍出的 f16 码本值与现有 `s_cbp` 完全相同，所以结果应逐 bit 一致。

为什么排第一：

- MoE 在 w4b 上约占 26%（09-28 Linux trace：lut 925+560 / 5720 ms per 8K）。kernel 即使只快
  ③ 的一半（−5%），16K 下整体也有 1.3%；与 ③ 持平就是 2.5–3%。
- **小 P 收益更大**：A 片段的解码按 tile 计，与该 tile 里有几行 token 无关（`live_tok_tiles`
  只跳过 WMMA，不跳过解码）。16K 时每个专家平均 320 行，解码被 5 个 tile 的 WMMA 摊薄；
  P=2048 时平均 40 行、只有 1 个 tile，解码几乎全露在外面。
- 原型可以直接在已经支持 `--hgn` 的 `tools/moe_lut_test.cu` 里加 `--opt`，与 ③ 同一套流程。
  进引擎门槛：up+down 合计 ≥5% 且逐 bit 一致。
- 日后 GGUF 的 IQ4_NL/IQ4_XS（同样是 16 项非线性码本）可以直接复用。

风险：v_perm 的选择子语义要对照 ISA 写对，选择子 ≥8 的分支最容易出错。用 CPU double 参考
加旧 kernel 逐 bit 对拍。

**状态（10-03，实测否决）**：原型已做并逐 bit 通过——`k_moe_lut` 加模板参 `kQ4Perm`
（默认 false，现有实例化逐 bit 不动；i4r/IQ 分支不受影响），寄存器码本为 r_lt/r_ht 8 个
v_perm 字节表，解码 = 每 word 2×(perm+bfi) 查表 + 8 perm 交织。harness `--q4perm 1`（仅
--hgn）设备端逐 bit 对拍 0 mismatch（hid/pairs 各全量），CPU double 参考 PASS。但 microbench
全面回退：P=16384 up −9.2% / down −5.3%，P=2048 −12.4%/−5.8%，P=512 −4.0%/−5.3%（合计
−7.8%/−10.0%/−4.5%，门槛 ≥+5%）。原因：占用率不变（105→112/113 VGPR，同分配桶；码本
进了 SGPR），每 stage 每线程多 ~300 VALU 在关键路径上——s_cbp 的 16 次 LDS 读原本完全藏
在 WMMA 之后，27 TFLOPS 下 kernel 是 WMMA/VALU 受限而不是 LDS 受限。i4r 算术解码能拿
−10.9% 是因为它每 4 权重只 ~6 条廉价 VALU，任意 16 项码本的 perm 查表做不到这个成本。
**未进引擎**；代码与 `--q4perm` 开关留存备查（同 GDEC_V2_PAIRS_F16 处理）。

### A2. 小 P 的 MoE 效率

前提：M1 显示 P≤2048 时 moe:up+down 的等效带宽明显低于 ~200 GB/s。

候选（按代价从低到高）：

1. **解码摊薄**：A1 先做。A1 之后小 P 的瓶颈可能会转到别处，所以 A2 要用 A1 之后的数据再评估。
2. **tile 粒度**：`k_moe_tiles`（10_ple_io.inc:769）固定 64 行一 tile，kernel 的 `BN` 模板参数
   已支持 16–64。小 P 时每个专家只有 10–40 行，BN=32 可以让 LDS 激活缓冲减半，提高 occupancy。
   但 B 片段读取不是瓶颈，收益要实测。gufo 按专家行数分 48/64/128 三档，方向相同。
3. **`moe_naive_max`=16 与 LUT 之间的过渡区**：P=17–128 时 LUT 每个 tile 只有 1–3 行。naive
   路径在 P=8 时带宽是 packed 路径的 2 倍（40_model.inc:1763 注释）。用 M1 的 P=64/128 两档
   看一下交叉点，可能只需要把默认值调到 32–64。只改常数，代价最小，可以随 M1 一起测。

验证：逐 bit（tile 切分不改变每行的累加顺序，需要核实一遍 `MoeTile.first` 的用法）+ pp。

### A3. 小 P 的 dense：直读量化权重

现状：P>8 时每个 dense gemm 都先把 q4cp/ht/q8g32 整张反量化到 `d_wbf16`，再交给 WMMA。
这一步要读 q4、写 bf16，GEMM 再读一遍 bf16，权重流量约为直读 q4 的 4.5 倍，而且按 chunk 计。
GUFO-GAP 实测 c2048 时 `k_dequant_q4cp_bf16` 每 8K 108 ms（≈27 ms/chunk）；v2 的 ht 是
71.5 ms/16K chunk。

做法：

- **q4cp / q8g32**：dense GEMM 的 A 加载器直接解码（与 MoE LUT 同一套）。最便宜的原型是把
  dense 权重当作"一个专家"喂给 `k_moe_lut`（q4cp dense 与专家同为 dtype 5），在 microbench 里
  看 P=512/2048 下与"反量化 + `k_gemm_wmma`"的对比。若能直接复用，工作量很小。
- **v2 ht（trellis）**：在 GEMM 加载器里解码要复杂得多，`k_ht_gemv` 的解码器只适合 gemv 布局。
  只在 M1 显示 v2 小 P 时 ht_deq >10% 才考虑；否则把 `gemv_multi` 的 P 上限往上推（P≤32 时
  gemv 直读可能仍然划算）就够了。
- 启用条件按 P 分流：大 P 继续走反量化 + WMMA（37.9 TF 已是天花板）。

收益：16K 下 <1%（v2 ht_deq 只占 0.7%）；P=512–2048 下 2–12%，取决于 M1。

**状态（10-03，w4b q4cp 原型已测，判定不做；v2 ht 只写评估）**：microbench
`moe_lut_test --hgn-dense W4B.hgn --tensor NAME --P N`（tools/moe_lut_test.cu
`dense_mode`：一张 dense q4cp 当单专家喂 `moe_lut_down`，对照 `k_dequant_q4cp_bf16`
+ `k_gemm_wmma` 引擎同款 dispatch 配置；x 同源，LUT 侧 f16、WMMA 侧 bf16；正确性
4 token×64 行 double 参考，误差按逐列乘积 rss 归一，lut ≤8.8e-4 / gemm ≤5.2e-3
全 PASS）。gfx1151 实测 ratio=(deq+gemm)/lut（>1 为 LUT 赢）：

| 形状（N,K） | P=512 | P=1024 | P=2048 | P=4096 | P=8192 |
|---|---|---|---|---|---|
| (2560,6144) d3 | **1.37** | **1.25** | **1.13** | 1.04 | 0.96 |
| (10240,2560) d9 | **1.51** | **1.11** | 0.92 | — | 0.78 |

LUT 平台在 ~28 TFLOPS（解码 VALU 封顶，不随 P 涨），k_gemm_wmma 大 P 到 35.7 TF；
交叉点 d3 ~P=4000、d9 ~P=1500。原始数据与口径注意见 logs/a3_dense_lut.md（deq 为
L2 热下界，冷 DRAM 更偏向 LUT 但不改交叉点量级；P<1024 时引擎 gemm() 实际走
hipBLASLt，40_model.inc:2050 的 `P >= 1024` 门，P=512 真实对照是 LUT ~28 TF vs
Lt 23–32 TF + deq，仅打平到小赢）。
**判定：不做**。收益窗口只有 P≤1024，生产 PREFILL_CHUNK=16384 在 0.78–0.96× 稳输，
P=512 按 Lt 口径也只是打平，不值得一条按 P 分流的 dense 路径。若未来主攻 ≤1K
chunk 的交互场景可重启（harness 与数据留存）。
v2 ht 侧：M1 实测 ht_deq 占比 P=256/512/1024/2048 = 19.5/13.3/10.7/4.4%
（logs/m1_m2_report.md），P≤1024 过 §3 的 >10% 门槛；但 trellis 解码进 WMMA A
加载器远比 q4cp 复杂（`k_ht_gemv` 解码器只适合 gemv 布局，WMMA 要 16×16 fragment
对齐的 LDS 暂存），且 ht 每权重解码更重，LUT 的 28 TF 天板只会更低。更现实的小 P
方向是把 `gemv_multi` 的 P 上限上推（P≤32 时 gemv 直读仍划算，只调常数）。列为
后续方向，本次不实现。

---

## 4. B 轨：所有 P 都受益的 kernel 项

### B1. down pairs 改 f16

- **状态（10-03，终判过线，v2 默认开进引擎）**：绝对 KLD（bf16_c512.kld，生产 env，--kld-chunks
  64）off=0.106993（精确复现基线）、on=**0.107705** ≤ 0.10699+0.002 **过线**（logs/kld_b1_abs_{off,on}.log）；
  默认开后复核：无 env=0.107705、`GDEC_V2_PAIRS_F16=0`=0.106993（logs/kld_b1_{default,optout}.log），
  decode chosen=248046 与 B4 一致。默认开已进引擎（40_model.inc，`GDEC_V2_PAIRS_F16=0` 退出），
  备份 build/gdec-win-B1.exe。pp 增益（128k@16384 背靠背，机器降级态口径）：10-02 +1.2~1.5%、
  同二进制相邻对 +0.75%（1177.6 vs 1168.8）、10-03 三联跑 +3.4%（1208.2 vs 1168.9，含单调热漂移
  放大，勿直接引用）。w4b 不受影响（pairs_f16 只在 v2 kI4R 路径）。
- w4b 同款：**（10-03，终判过线，默认开进引擎）**。实现：`moe_lut_down` 走
  `k_moe_lut<kQ4CP,…,kF16Pairs>` 写 f16 pairs，三个 reduce kernel 模板化 `<int K, class PT>`
  读 f16（`__half2float` 精确拓宽，fp32 累加不变，与 v2 同款）；开关 `GDEC_W4B_PAIRS_F16`
  （默认开，=0 回退），`d_guvb` 在 `guvb_f16_env()` 门控下减半（v2 flag 同开 + q4w LUT 路径 +
  topk=10 + mid%64==0，排除 naive/atomic/host-route/ordered/untiled/GGUF 全部回退路径）。
  验证：c512 KLD off=0.163530、on=**0.162452**（门 off+0.002 过线且反而更低；默认开后无 env
  精确复现 0.162452）；v2 回归 0.107705 精确复现（改动对 v2 零影响）；pp 严格空闲背靠背
  （128k@16384）on **1172.0** vs off 1168.2 = **+0.3%**，KPROF 腿 1168.8 vs 1151.9 = +1.5%；
  KPROF moe:reduce **−36%**（302.0 vs 468.7 ms/16K）、moe:down **−12%**（946.1 vs 1077.4）、
  segsum **−1.4%**（−205 ms；gdn +3% 方向两次测量一致，疑为频率/功耗重分配）；devarena
  90.68 vs 91.46 GiB（**−0.78 GiB**，无 hipMalloc fallback）。第三读者审计：d_guvb 的
  GGUF/moe_lt/tiled-w4 读者均被 gate 排除，naive gemv 部分积与共享专家 scratch 每 token 恰为
  8·mid 字节与 f16 pairs 相等，减半安全。注：早前一次 −0.5% pp 为测量污染（并行 grep 期间
  采集），空闲复测翻正。
- 注意：09 月的 `GDEC_MOE_PAIRS_BF16` 打平，是因为当时 reduce 不是瓶颈；而且 bf16 精度比 f16
  更差，不要拿它当先例否决这条。

### B2. HC 链（15.3%，粗估下限 1.2–1.5 s，实测 1.67 s/16K）

每个 HC 点（每层 2 个，共 96 个）的流程：

1. `k_gr_scatter_norm_inj_bf16<4>`：读 R、写 R、写 Rhat，并算 inject 部分和
2. mix-down GEMM（N=320，K=10240，读 Rhat bf16）
3. `k_silu_scale`
4. `hc_up_fused`（`k_gemm_wmma<128,128,…,Epi=1>`，N=10240，K=320），写 `d_xb` f32 + `d_xbf16`

带宽与 GEMM 各自的下限约 0.9 s 和 0.6 s，两者部分重叠，因此剩余空间约 0.2–0.45 s（2–4%）。
三个子项：

1. **MoE reduce 并进 gr_write**（~0.7%）：`k_v2_reduce_rot` / `k_moe_reduce_pw_sg` 把结果写成
   `d_xb` f32，紧接着的 HC 写回又读一遍。合并后每层省 P×2560×4×2 字节。运算顺序可以保持不变，
   应能逐 bit 一致。注意 v2 与 w4b 是两个 reduce kernel，都要改。

   **状态（10-03，已否决，默认关留存）**：已实现进引擎，`GDEC_HC_FUSE_WRITE=1` 开启（同
   A1/B1 先例）。v2 `k_v2_reduce_rot_hc`（28_kernels_hgn_v2.inc）、w4b
   `k_moe_reduce_pw_sg_hc`（24_kernels_moe_lt.inc）= 原 reduce + 尾部 4 分支 bf16 R RMW
   （表达式与 `k_gr_write_b_hc_bf16` 逐字一致）；`k_gr_scatter_norm_inj_bf16` 加模板参
   `<int T, bool Write>`，Write=false 跳过 R 写回只做 norm/Rhat/pout（if constexpr，默认
   路径零开销；初版运行时 `if(yg)` 分支曾让 scatter 慢 ~10%，hc2 +83.5 ms，已模板化消除）；
   主机侧 `gr_write_prep_b` + write_done 管线（40_model.inc），仅在 LUT sg_fuse 路径融合。
   验证：ktest 8 项逐 bit A/B 0 mismatch（hcw_reduce_pw_sg / hcw_v2_reduce_rot /
   hcw_scat_ypath_{R,Rhat,pout} / hcw_scat_split_{R,Rhat,pout}，logs/b21_ktest_final.log）；
   v2 KLD 自 A/B mean_kld=−0.000000 same_top=100.000（logs/kld_b21_v2.log），w4b
   mean_kld=0.000000 same_top=100.000（logs/kld_b21_w4b.log；w4b 基线
   logs/kld_w4b_before.kld 是 B4 二进制补存——B4 只动 v2 kMid，w4b 不走 kMid）；最终
   二进制（模板化+默认关）上 off/on 两次 KLD 均逐 bit（logs/kld_b21final_v2_{off,on}.log）。
   否决依据（全部实测）：KPROF 32k@16384 稳态 segsum 10665.7→10848.3 ms（+182.6，+1.7%）：
   moe:reduce +138.1（+31%）、hc −51.3、hc2 +83.5（上述分支开销，已修）、hc3 −7.6
   （logs/b4_v2_32k_kprof.log vs logs/b21v2kprof.log）；pp v2 128k@16384 稳态 avg 1369.8 vs
   B4 基线 1443.7（**背靠背口径 −5.1%**；对 1454.8 那跑为 −5.8%，1454.8 已判定属运行间
   漂移——见 B4 状态段；两 run 逐 chunk 同斜率，chunk 对 chunk 一致 −5~6%）。
   **流量模型修正**：上面 ~0.7% 的估算漏算了 `gr_write_read_b` 的 scatter 本就把写回与
   norm 融合在同一次 R 读里（写后值在寄存器里直接 norm）。46/48 个 mlp 站点走 deferred
   路径：写回挪进 reduce 后 reduce 读+写 R（671 MB/层），scatter 为 norm 重读 R
   （335.5 MB/层），新增 R 读恰好抵消省下的 y 往返（2×167.8 MB/层），理论净零；实测
   reduce 侧 RMW 效率更差 → 净回退。仅 2/48 个 flush 站点（独立 `k_gr_write_b_hc_bf16`）
   能真省，~0.03% 不值得。教训：HC 链优化先做流量闭合核算再上 microbench。
   最终二进制回归（默认关 ≡ B4）：pp v2 1439.8 vs B4 同机背靠背 1443.7（−0.27%，逐 chunk
   ≤0.7% 噪声，logs/b21final.log / logs/b4rerun.log；B4 当天早些时候的另一跑 1454.8 属
   运行间漂移）；pp w4b 1410.4 vs 基线 1412.4（−0.14%，logs/b21w4b.log）；w4b 2k smoke
   decode ids 与 B4 一致（logs/b21w4bsmoke.log）。
2. **免 Rhat 落盘**（~1.3%）：mix-down GEMM 的 A 加载器读 R 并现场归一化（rms 用一个小 kernel
   先算出，或从 scatter_norm 里顺带写出 P×4 个 scale）。这样省掉 Rhat 的一写一读（16K 时
   每点 2×335 MB）。代价是 GEMM 加载器多做乘法，且归一化后转 bf16 的时机不同，可能**非逐 bit**，
   需过 KLD。动手前先核实 Rhat 是否还有别的读者（gemm() 里 `residual_ready` 那条路径会用
   `d_Rhatbf16`）。

   **状态（10-03，已否决，默认关留存）**：已实现进引擎，`GDEC_HC_NORHAT=1` 显式开启（同
   B2-1 先例）。Rhat 读者全核实：mix-down GEMM、hc_up_fused Epi=1 的 Rh、gr_read_b 后的
   inject GEMM（仅 ~3 站/chunk，**gr_read_b 不动**，仅 gr_write_read_b 的 93/96 站免 Rhat）。
   scatter/rmsnorm 只写 P×4 个 f32 inv scales 到新缓冲 d_hcsc；mix-down 走新 Epi=2
   （A staging 现场归一化，与 rmsnorm 逐字同表达式 → A tile 逐 bit）；hc_up Epi=1 加
   Sc/Hw 可选参现场乘 (rv·sc)·(1+Hw)（全链唯一数值差）。门控 hc_norhat（40_model.inc）
   与 hc_up_fused 条件逐一对应，回退即 fatal 防读 stale Rhat。验证：ktest ALL PASS
   （logs/b22_ktest.log）；v2 c8192 KLD 0.081293 vs 基线 0.082492（**还略好**，少一次
   Rhat bf16 中间舍入；logs/kld_b22_c8192_{on,base}.log，注意 c8192 需 `--maxctx 8192`，
   默认 maxctx=4096 会 rc=2）；v2 c512 绝对 0.107705（P=512 不触发，等于 B1-on 终值）；
   w4b c512 0.000000、w4b c8192 0.121676 vs 0.121507（+0.000169）。否决依据（实测）：
   pp v2 128k@16384 背靠背 on 1175.7 vs off 1216.4 = **−3.3%**（机器当时处于 −19% 降级态，
   但背靠背口径有效；logs/b22_norhat_{on,off}.log）；KPROF 32k 稳态 hc+hc2 段 2271 vs
   1978 ms/chunk（**+293 ms**）——Epi=2 staging 归一化的 VALU + Sc/Hw 额外读超过省下的
   Rhat 落盘带宽（logs/b22_kprof_{on,off}.log）。教训同 B2-1：省的是 DRAM 流量，花的是
   执行单元时间，本机瓶颈在后者。
3. **up-fused GEMM 效率**：只有 ~23 TF。K=320 只有 10 个 K-step，序言、尾声和 Epi=1 的双写
   （f32+bf16，P×10240）占比高。可以试 128×256 tile（09-28 在 8192 下测过，2.80 vs 2.33 ms，
   更慢；但当时还没有 inject 融合，16384 下可再测一次）或 split 写回。上限约 150 ms/16K（1.4%），
   把握不大，排在 1、2 之后。

   **状态（10-03，microbench 复测，tile 方向否决）**：`hcmix_proto`（tools/hcmix_proto.cu，
   gemm_wmma_kernel.inc 已从当前 22 重新 sed 提取，HCMIX PROTO: PASS）在 P=16384 复测：
   引擎配置 `<128,128,4,2,2,4,32,256,1>` gm2 = **4.711 ms（22.8 TF）**，128×256 tile
   `<128,256,2,4,4,4,32,256,1>` gm1 = **5.793 ms（慢 23%）**；8192 下 2.931 vs 2.402 ms，
   与 09-28 结论一致。同扫 13 种配置（64×128 / 128×64 / 64×256 / KST=64 / 32×64 / gm1,2）
   无一胜过现配置，全部与未融合对拍逐 bit 0 diff（logs/b23_hcmix_16384.log）。tile 方向
   两次实测皆输 → **否决**；split 写回未试，剩余上限 ~150 ms/16K 且把握不大，不再投入。

### B3. dense k1 形状 split-K / stream-K

`gdn:oproj` 646 + `qsa:oproj` 209 = 855 ms/16K，形状 (N=2560, K=6144)，只有 ~29 TF，而 d9 形状
能到 35+。gemm_lt 已全扫（d9/gm=4、k1/gm=1 已最优）。PREFILL.md §11.1 写明"N=2560 形状
persistent + K 拆分（未试）"。

**状态（10-03，过线，默认开进引擎）**：split-2（K 6144→2×3072）已实现。proto
（tools/b3_splitk_proto.cu，logs/b3_splitk3.log，门槛"切片合计 ~35 TF 等效留 1 ms combine
余量"）：P=16384 base 31.4 TF → 切片 36.5 TF-equiv（x1.16）**过线**；关键分解实验：**收益全部
来自 X（激活）连续切片**——contX+stridedW 35.4 TF-equiv（x1.13）保住绝大部分，而全 strided
免重排版 x0.84、stridedX+contW x0.91 皆输 → W 维持原布局 strided 读（零权重内存代价），X 每次
调用经 `k_splitk_repack2` 重排（+0.23 ms）。引擎实现：k_gemm_wmma 加 ldA/ldB 行距参（0=K，
旧调用不变）+ Epi==3 原子加尾声（每元素两次 launch 各属唯一 CTA → 恰好一次 RMW，跨 run 逐
bit 确定）；gemm() k1 路由在 P≥8192 时走 split-2（P=4096 proto 仅 x1.01，不走），slice0 写
Y、slice1 原子加；新缓冲 d_skx（BP×6144 bf16，层边界即死不入 WS 小拷贝；devarena_estimate
已同步）。门控 `GDEC_GEMM_SPLITK=0` 退出（gemm_splitk_on，40_model.inc）。修复记录：首版
slice0 漏传 ldB（W 被当连续 3072 行距读 → KLD 9.56 全烂），已修。
验证：ktest ALL PASS；v2 c8192 KLD 0.083512 vs off 0.082492（+0.00102 ≤ 0.002；
logs/kld_b3_c8192_{on,off}.log）；w4b c8192 0.122617 vs 0.121507（+0.00111；
logs/kld_b3_w4b_{on,off}.log）；v2 c512 绝对 0.107705 不变（P=512 不触发）。pp v2
128k@16384 背靠背 on 1223.9 vs off 1208.2 = **+1.3%**（logs/b3_pp_{on,off}.log）；32k
KPROF 对 +1.8%（1276.6 vs 1253.9），gdn:oproj+qsa:oproj 1631→1267 ms/chunk（−364 ms；
logs/b3_kprof_{on,off}.log）。三联跑 decode chosen=248046 一致。gm={2,4,8} 全 K 变体
x0.66-0.88 全输（proto 顺带扫）。128K 布局 +14% 放置漂移：pp A/B 均固定 maxctx 同脚本
背靠背。

原方案备忘：K=6144 拆 2 段，每段 3072，f32 部分和用原子加或二次 reduce。N=2560 只有 20 个
N-tile，拆 K 后并行度翻倍。非逐 bit（K 累加顺序变），已过 KLD。

### B4. kMid occupancy（v2）

VGPR 169 → 1 CTA/CU；pass 0/1 落 LDS plane 时有 16-way bank conflict。plane 行距加 1 个 half2
padding 可消掉冲突（逐 bit，零风险，先做）。VGPR 降到 ≤128 能让 2 CTA/CU，但①②的经验表明
这个 kernel 对寄存器很敏感，先用 `-Rpass-analysis` 看看大头在哪再决定。上限约 1%。

**状态（10-03，已完成进引擎）**：plane 行距 BM→BM+2 halves（kMidRow），写侧 16-way bank
conflict 消除。microbench（`moe_lut_test --v2 --mid 1`）逐 bit 0/104857600 mismatch，fused
up+mid 37.67→36.78 ms @P=16384（**+2.4%**；对比组：零 LDS 增长的 rotate-by-2t swizzle 变体
只 +1.8%，padding 胜出，swizzle 未留码）。 VGPR 实测 111（不是旧的 169），但 2 CTA/CU 被
LDS 结构性封死（2×16.6 KB plane，64 KB/CU），VGPR 一项无意义——第二步不做。引擎验证四项
全过：KLD 自 A/B mean_kld=−0.000000、same_top=100.000（与历史逐 bit 接受行逐字节一致）；
pp 128k@16384 稳态 **1454.8 vs 基线 1439.0 = +1.1%**（n=7 剔前 2；同日背靠背复测
B4=1443.7（logs/b4rerun.log），运行间漂移 ~0.8%，pp 增益应读作 +0.3~1.1% 区间——硬证据
以 microbench +2.4% 与 KPROF moe:up −1.8% 为准）；KPROF moe:up 段
1704.5→1673.9 ms/16K（−1.8%）；w4b smoke PASS（w4b 不走 kMid）。日志：logs/kld_v2_b4.log、
logs/b4_v2_128k.log、logs/b4_v2_32k_kprof.log、logs/b4_w4b_smoke.log。

### B5. rot_in 并进 norm（v2，0.5%）

维持最低优先级，见 HANDOFF-V2-PREFILL.md §5。

**状态（10-03，分析后否决，未做原型）**：`k_v2_rot` 本身已在带宽顶（57 ms/16K = 读 f32
168 MB + 写 f16 84 MB × 48 层 ≈ 212 GB/s），唯一省法是生产者融合（省 f32 重读 ≈ 38 ms/16K
≈ 0.3%）。但现行配置（GR_BF16+GEMM_WMMA，P≥1024）下 d_xb 的生产者是 `k_gemm_wmma` Epi=1
的 epilogue：W 行置换后**每 CTA 只覆盖 32 个输出通道**（(n0>>2)+(wp0>>2)+cl，BP=128 =
4 分支 × 32 通道），而 rot_in 的 FWHT 需要完整 128 通道组——跨 4 个 CTA，不重构 tile/置换
（BP=512 级改动，重调参）做不了，风险/收益完全不匹配。小 P 路径 `k_gr_combine_b_hc_bf16`
（一 token 一 block，天然可行）在 P≥1024 不走；为它关掉 hc_up_fused 会赔掉约 250 ms/16K 的
HC 融合收益，净亏。读 d_xbf16 代替 d_xb 省一半读流量但不逐 bit，超出本轮口径。结论：**否决**，
维持原状；若日后 dense gemm 的 HC epilogue 重构再顺带重新评估。

### B6. 其余 gufo 式尾声融合：已基本做完

09-28 起已做完 convl2、sigmoid gate、shared 并进 reduce、HC inject。剩下的独立 kernel 都已在
带宽极限附近，融合后每项只能省一次中间张量的往返：`gdn:norm` 184（86% 带宽）、`qsa:prep` 84、
`qsa:gate` 84 ms/16K。估计合计 <0.5%。gatednorm 并入 gdn_fused 已因 LDS 不够否决。**不建议再做。**

---

## 5. C 轨：长上下文（32K–256K）

128K 稳态里随 base 增长的只有两处：`qsa:idx` 381→1032 和 `qsa:flash` 940→1371 ms/16K。

### C1. indexer 打分 bf16 WMMA（128K ~3.5%）

- t64 打分只有 8.7 TF（fp32 FMA）；占用率修法已否决（ST=8/4 只 +4~5%）。换成 bf16 WMMA，
  按 dense GEMM 的经验能到 2× 以上。128K 省 ~400 ms/16K。
- 顺带把 score 矩阵改 bf16 存储，select 的读写流量减半。select_2p 已在 ~220 GB/s 流率上，
  这正是它唯一能变快的方式。
- 属于数学改动：top-512 选块在 bf16 分数下可能有并列与换位。**必须用 ≥32K 的 KLD 把关**（M3），
  c512 测不到（2051 以内全选）。另外看一下 needle 检索（YARN-512K-RESULTS.md 的方法）作为第二
  判据，选块换位最先影响的是长程检索。
- 先在 `tools/index_fast_test.cu` 的框架里做原型，统计与 fp32 选块的重合率（目标 ≥99.5%）。
  重合率不过关就不必去跑 KLD。

  **状态（10-03，原型完成，重合率未过 99.5% 门 → 按预案停在原型，未进引擎）**：
  `k_index_scores_bf16`（09_kernels_index.inc：16 token×128 key/CTA，K=128 全 staging，
  8 warp 2×4、wmma 16x16x16 bf16，lane 内 8 行恰好 2 token×4 head → ReLU+head 求和
  全在 lane 内）+ select_rs/select_2p 模板化支持 bf16 读（ST=float 默认，旧路径逐 bit
  不变；bf16→f32 只是 bits<<16，排序语义不变）。实测（logs/c1_proto1.log）：
  - **性能达标**：打分 17.3-17.8 TF = t64 的 **x1.96-2.04**（128K/256K last batch）；
    引擎侧未动，qsa:idx 的打分部分本来能砍一半。
  - **设备 select 精确**：bf16 score 上 select_rs/select_2p 与 host 全序（score desc、
    并列取小 id）逐行一致，无 DEVICE-SELECT-DIFF。
  - **重合率不过门**：vs fp32 打分+选块，mean 0.9926-0.9994——8K 0.9994 过，48K/32K
    0.9957 勉强，**128K 0.9941、256K 0.9926、odd-nb 0.9944 均 < 0.995**。分数 relL2
    ≈0.0024（bf16 输入舍入主导），在第 512/513 名的天然小间隙上换 1-2% 是 bf16 尾数
    （8 bit）的固有极限，不是实现瑕疵。随 nb 增大单调变差。
  - 另：select bf16 反而 x0.87 慢（2B 散读未向量化，pass-1 直加载效率腰斩；可修但因
    重合率否决而无意义）。
  - **后续若要重提**：换 f16（尾数 11 bit，relL2 估 ~0.0003，重合率估 >99.9%；gfx1151
    上 f16/bf16 wmma 同速，score 同样能 f16 存）。这是唯一没被本次数据否决的变体。
  - 代码全部留存（kernel + harness + 本段数据），引擎零改动。

  **状态（10-03 晚，f16 变体：原型过门 → 已进引擎，默认开，`GDEC_INDEX_F16=0` 回退）**：
  bf16 被否决时预判的 f16 变体（尾数 11 bit）完全兑现。
  - **动态范围分析（结论）**：输入是 RMS-normed(+rope) 的 indexer q/k（`index_norm_rope`：
    value·rsqrt(Σ/128+eps)·(1+w)，64 维 rotary），量级 O(1-10)，f16 上限 65504 绰绰有余；
    点积 K=128 的 wmma 累加器本来就是 f32，只有输入与最终 score 舍入；score = 4 head
    max(0,·) 求和 ×1/√128 ≤ 数百，不溢出。f16 尾数 11 bit（rel step 2^-11≈4.9e-4）是
    bf16 的 8 倍细：实测 relL2 0.000298 vs bf16 0.00239，与预估 ~0.0003 一致。
    排序语义：score 经 ReLU 非负，`__half2float` 精确转 f32（f16 ⊂ f32）后
    histogram/radix 代码与 fp32 路径完全同构，零风险。
  - **原型（logs/c1_f16_proto1.log）重合率 mean/min（门 ≥0.995）**：8K first 0.99992/0.9980；
    8K ragged 0.99966/0.9961；small17 0.99977/0.9980；count16 0.99939/0.9980；
    48K 0.99937/0.9961；32K 0.99937/0.9961；odd nb 0.99916/0.9941；128K 0.99912/0.9941；
    256K 0.99886/0.9941。全部 mean ≥0.99886（门的 ~4 倍余量）；min 行 0.9941 略低于
    0.995 但 mean 口径过门（min 是单行最坏，1024 行中个别行在第 512/513 名间隙换位）。
    设备 select（ST=__half）与 host 全序逐行一致，零 DEVICE-SELECT-DIFF。
  - **microbench（128K/256K last batch）**：f16 打分 16.66/17.66 TF = t64 的 **x1.93/x2.00**
    （与 bf16 同速，预测兑现）；select f16 x0.87/x0.83 反而慢（2B 散读未向量化，与 bf16
    同病）。净账（128K/batch）：score 3.96→2.06 ms 省 1.9，select 亏 ~0.18，净省
    ~1.7 ms/batch。**select 向量化（uint4 一次 8 个 f16 + 对齐 peel）是已识别的后续
    优化，本轮未做**。
  - **引擎实现**：d_iqh/d_ikh/d_ikgh 三个 f16 twin 数组（分配门控 = !GDEC_INDEX_F16=0 &&
    SCORE64 && SEL2P && !OLDSEL，与 30_host_util arena 估算同步）；`index_norm_rope`
    加 out16 尾参（__float2half_rn RNE），k_index_q/k_index_pool/k_index_append 全部
    写 twin（decode 也写 → 多轮 prefill 不会读到陈旧 twin）；score 复用 d_iscores 内存
    按 f16 存；n≤8192 走 `select_rs<true,__half>`，否则 `select_2p<__half>`；paged
    gather 加 `k_ik_gather16`。**twin 一致性三处修复**：① kv_page_regions_k 把 d_ikh
    纳入页枚举（kv_cow 拷贝/调试清零/guard 全覆盖）；② k_index_append（decode 完成
    block）写 twin；③ kvsnap restore 后用新 kernel `k_ik_twin_build` 从 fp32 重建
    [0, n/4) 的 twin（twin 是派生态不进快照格式，BTV 重建同款模式）。WS（并发小
    工作区）经 GDEC_WS_FIELDS 宏自动覆盖 d_iqh。fp32 路径零改动（ST=float 默认模板
    参，ktest ALL PASS）。
  - **引擎验证（同机相邻口径；机器降级态 pp ~1220，历史 1443.7）**：
    KLD（对 data/kld/bf16_c8192.kld，--maxctx 8192，门 = off+0.002）：v2 on **0.082705**
    vs off 0.083512（−0.0008，反而更低：f16 舍入恰好更贴近 bf16 参考）；w4b on
    **0.122480** vs off 0.122617；c512 **0.107705 逐位精确**（<2051 f16 完全不激活的
    对照）。off 值与 B3-on 基线（0.083512/0.122617）精确复现。twin 修复前后 KLD 逐位
    不变（on2/c512_2 复跑确认）。pp 128k@16384 背靠背：on **1222.5** vs off 1205.6
    tok/s 稳态 avg（**+1.4%**，n=7；min 带 1186-1261 vs 1160-1260），两边 step 0
    chosen=248046 一致。KPROF 32k：qsa:idx on 316.3 vs off 340.3 ms/16K（−24 ms，
    −7%；32K 的 nb 小，128K 下绝对收益更大），segsum −280 ms/chunk，稳态 tok/s
    1315.5 vs 1281.2（+2.7%，单样本）。**KPROF 128k**：qsa:idx avg on **464.0** vs
    off **619.8** ms/16K（−25%），末 chunk（nb≈32K）on 732.7 vs off 1040.6（−308 ms，
    −30%；off 精确复现历史 ~1032）。账：128K 每 chunk 打分 ~7.15 TFLOP（13 层 ×
    16K×4 head × 32K block × 128K-dim），t64@8.7TF ≈ 820 ms → f16@16.7TF ≈ 430 ms，
    与实测吻合；qsa:idx 里另有 ~300 ms 底（iproj GEMM ~130 + select ~130-150 +
    norm/rope/gather）不吃这个优化，所以到不了 "~500 以下" 的预期。稳态 tok/s
    on 1212.3 vs off 1194.5（+1.5%，第二组相邻对照，与第一组 +1.4% 一致）。w4b 2k smoke：ids 与 logs/b4_w4b_smoke.log
    逐位一致（chosen=248068；2k 下 f16 不激活，纯回归）。needle（tools/c1f16_needle.py，
    64K 冷 prefill 经 f16 路径，10/50/90% 深度）：**3/3 命中**。
  - 日志：kld_c1f16_{c8192_on,off,c512,w4b_c8192_on,off}.log、kld_c1f16_{c8192_on2,
    c512_2}.log（twin 修复后复跑）、c1f16_{on,off}.log（pp 128k）、c1f16_kprof_{on,off}.log、
    c1f16_kprof128_{on,off}.log（KPROF 128k）、c1f16_w4b_smoke.log、c1f16_needle_rerun.log。

### C2. gdn:scan 深流水（≤3.5%，数天）

`k_gdn_fused` 受延迟和占用率限制（48 CTA / 20 WGP = 3 波，波效率 ~60%）；LDS 已用到 57/64 KB，
只能靠寄存器倒手。这是 GDN 维度唯一没被否决的形态（否决清单见 §8），正确性 harness 复用
`tools/gdn_fused_proto.cu`。收益不确定，排在 C1 之后。

### C3. 不建议：qsa:flash

已在 DRAM 流率极限（L2 吸收了 ~3/4 的邻近重叠）。再降只能降 KV 精度（fp8 KV），质量风险
远大于 128K 时的 ~2%。halogen 也是 bf16 KV。

---

## 6. D 轨：少做 prefill（缓存命中）

这几条不让 kernel 变快，但在 agent / 多轮场景里，它们省的是整段几千 token 的重算，用户感知
往往比上面任何一项都大。全部来自 09-26 前缀复用修复时记下的遗留（aca5987 之后未动）。

### D1. decode 吐出 `<tool_call>` / `</think>` 时存 rckpt

现象：API 的 TokenCache 把切点放在 `<tool_call>`（token 248058）之后，但引擎在那里没有
checkpoint。于是回退到上一次 prompt 结尾（rckpt 最小 4096、最多 8 个），**每个 agent 回合都把
上一条回复整段重新 prefill 一遍**。工具调用参数的重新序列化还可能让回放在回复中途就分歧。

做法：decode 循环里一旦采样到这两个 token，立即存一个文本 rckpt（49_rckpt）。`</think>` 也要，
因为模板常在多轮时裁掉 think 段。需要处理：

- rckpt 数量上限与淘汰策略；
- MTP 投机时 token 是成批接受的，要在接受之后、对应位置上存；
- 并发 slot 下的显存预算。

验证：在 `tools/tcache_verify.sh` 里加一个 agent 回放场景（两轮工具调用），要求第二轮的
`cached` 覆盖到 `<tool_call>` 处。

### D2. TokenCache 落盘

TokenCache 只在 API 进程内存里（4M token / 64 条）。服务重启或条目被淘汰后，老的长对话会按
规范 BPE 重新编码，和 kvsnap 里存的生成 id 对不上，于是近乎全量重算。把 TokenCache 和 kvsnap
一起落盘（同一个指纹），重启后长对话首轮即可命中。

### D3.（低）长 decode 期间定期存 rckpt

09-26 的 Fix 2，未做。D1 做完后价值更低。

---

## 7. 配置与内存

- **E1**：Windows 按 `devarena_estimate`（chunk 16384）+ slack ≤ cap 自动选 16384。注意
  `devarena_estimate` 内部调用了 `eff_maxbatch`，不能在后者里直接调（会递归），要在
  `devarena_init` 之前先算一次，写进一个全局 override。256K serve 用 16384（粗算 E≈91.7 GiB）
  还没实机验证，上线前要看 `devarena: requesting` 那一行和有没有 `fall back to hipMalloc`。
- 小 P 方向（A1–A3）做完后**重新扫一次 chunk**：如果 4096/8192 与 16384 的差距缩小到 1–2%，
  Windows 就可以用小 chunk 换回 8–12 GiB arena。这比 E1 更根本。
- GGUF 在 Windows 上没有移植，arena 也放不下（~102 GiB），不在本轮范围。Linux 上 GGUF 能吃到
  A3（Q8_0 dense 也是每次调用都反量化）、B2、B3、C 轨，A1 不适用（GGUF 专家走的是 gufo 移植
  kernel，已经是魔数解码）。

---

## 8. 已否决总表（不要再提）

| 项 | 结论 | 出处 |
|---|---|---|
| A1：q4cp perm 寄存器查表 | bit-exact 但 −4~9% 回退（VALU 关键路径，LDS 读本就藏在 WMMA 后；occupancy 不变） | 本文 §3 A1；tools/moe_lut_test.cu --q4perm 留存 |
| A3：dense q4cp 直读 LUT GEMM | LUT 平台 ~28 TF 封顶，只在 P≤1024 赢；生产 16K 稳输，P<1024 对 Lt 口径仅打平 | 本文 §3 A3；--hgn-dense 留存 |
| B2-1：MoE reduce 并进 gr_write | 逐 bit 正确但净回退（deferred 路径新增 R 读恰好抵消省的 y 往返，理论净零） | 本文 §4 B2-1；`GDEC_HC_FUSE_WRITE` 默认关留存 |
| B2-2：免 Rhat 落盘（Epi=2 现场归一化） | KLD 全过（c8192 还略好）但 pp −3.3%、hc+hc2 +293 ms/chunk：VALU/额外读 > 省的 DRAM 流量 | 本文 §4 B2-2；`GDEC_HC_NORHAT` 默认关留存 |
| B3 split-K 的 strided 免重排版 / 全 K gm={2,4,8} | strided x0.84（收益全在 X 连续化）、gm 变体 x0.66-0.88 | 本文 §4 B3；logs/b3_splitk3.log |
| C1：indexer 打分 bf16 WMMA + bf16 score | kernel x2 达标（17.5 TF）但选块重合率 0.9926-0.9944 < 0.995 门（128K/256K），bf16 尾数固有极限；**f16 变体过线，默认开（见 §5 C1，非本表否决项）** | 本文 §5 C1；logs/c1_proto1.log |
| B2-3：hc_up_fused 128×256 tile | 16384 下 5.79 vs 4.71 ms（慢 23%），8192 下与 09-28 一致再输；13 配置无胜者 | 本文 §4 B2-3；logs/b23_hcmix_16384.log |
| B5：rot_in 并进 norm | 生产者 Epi=1 epilogue 每 CTA 只 32 通道，FWHT 要 128，跨 CTA 不可行 | 本文 §4 B5 |
| PLE 预取提前到 tokenize 后 | 661 ms 系冷页缓存一次性离群；稳定态首 chunk ~P×0.125 ms（v2/w4b 同一张表） | 本文 §2 M1 ④ 复核 |
| int8 / iu8 WMMA | gfx1151 上 iu8 与 f16 同速；iu4 需 4bit 激活，质量不可接受 | V2-PREFILL §4 |
| 所有双流 overlap（LT_OVL、GDN 窗口/pipe、GR_SCAT4 等） | 执行单元无空闲，同向否决 | PREFILL §11.3/7 |
| dense GEMM 三缓冲/预取加深、gemm_lt 全扫 | spill / 已最优 | PREFILL §11.1、V2-PREFILL |
| GDN fp16 WMMA、strip fp16 ws、fused3 半 chunk 流水、劈块 | 否决，GDN 只剩 C2 | PREFILL §11.4/5、V2-PREFILL |
| gatednorm 并入 gdn_fused | LDS 不够 | PREFILL-NEXT |
| GR 去重读 | 否决 | PREFILL §11.7 |
| t64 短 staging（占用率） | 只 +4~5% | V2-PREFILL |
| select_2p 调参、fused FEAT=8 | 已在流率上 / 整体 0.3% | V2-PREFILL |
| idx hist 融进打分 | 128K 1.3%，估算误差 ±2×，可能净负 | V2-PREFILL |
| k_moe_lut BK=4、每 wave 2 个 A 片段 | VGPR 169/156，occupancy 回退 | V2-PREFILL §4 |
| kMid 原方案 BM=256 | VGPR 101→156，up +8.5% | V2-PREFILL §3 |
| 旧 MoE WMMA 原型（09-21） | 已被 LUT kernel 取代 | PREFILL §11.2 |
| MoE f16 输入（取自 hc_up_fused） | buffer 冲突，~17 ms | PREFILL-NEXT |
| `GDEC_QSA_UNION` | 端到端无收益 | PREFILL-NEXT |
| `GDEC_MOE_ATOMIC` | −16% | PREFILL 文首 |
| oproj 128K 漂移的 pad 修法 | 物理放置抽签，无通用偏移 | V2-PREFILL |
| ht dense 常驻 bf16 | 多占数 GiB，违背 v2 初衷 | V2-PREFILL §6 |
| prefill 走 hipGraph | 未经测量不做；只有 M1 显示 GPU 空转 >5% 才重提 | 本文 M1 |

---

## 9. 验证口径（沿用并补充）

- **逐 bit 改动**（A1、A2、B2-1、B4）：microbench 与旧 kernel 设备端逐位对拍；引擎自 A/B KLD
  （`logs/kld_v2_before.kld`）要求 mean_kld ≈ −0.000000、same_top=100.000；**w4b 也要跑一遍**
  （A1 只动 q4cp 分支，v2 必须逐位不变）。
- **非逐 bit 改动**（B1、B2-2、B3）：c512 绝对门槛，v2 0.10699+0.002；w4b 以同机改前值 +0.002。
  注意 c512 的 P=512 **不触发** hc_norhat（P≥1024 门）与 split-K（P≥8192 门）——这两条路径的
  真实覆盖靠 bf16_c8192.kld（需 `--maxctx 8192`，默认 4096 会 rc=2），门槛同机改前/off 值 +0.002。
- **C1**：长上下文 KLD（M3）+ 选块重合率 + needle。
- **pp**：同 maxctx、同 chunk；128K 取 n≥7 个稳态 chunk，剔除前 2 个；小 P 用 M1 脚本，报
  P=512/2048 两档。
- **D1/D2**：`tools/tcache_verify.sh` 扩展场景，结尾 PASS/FAIL；`api_regression_test` 90/90。
- microbench 必须在 GPU 空闲时跑（曾被后台 KLD 污染出假数据）。
- **测量纪律（10-03 立规）**：任何 GPU 测量（pp/KPROF/microbench/KLD）期间，禁止并行编译、
  禁止并行 grep/大量文件操作、禁止第二个 GPU 进程；编译与测量严格串行。10-03 一次并行 grep
  期间采集的 B1-w4b pp/KPROF 把墙钟测成 −0.5%、段级数据全偏，严格空闲复测翻正为 +0.3%。
- **机器降级态（10-03 判定）**：系统安静下探针 1228.7 tok/s（v2 128k@16384），未回到 10-02
  的 1443.7 → 是真降级（热/硬件/驱动），**不是**测量期 CPU 抢功耗。相邻背靠背口径仍是正确
  方法论，跨日绝对值比较须先跑探针校准。
- 对外报收益时区分「kernel 收益」与「chunk/内存换来的收益」。

---

## 10. 下一步方向（10-03 记录，用户拍板全部要做）

10-03 收尾时用户确认以下剩余方向都要做。文档里已有零散描述的，此处收拢指针与关键数据；
每条标注前置依赖、验证口径、预估收益。

1. **D1 rckpt 缓存命中**（规格见 §6 D1）：decode 吐出 `<tool_call>` / `</think>` 时立即存文本
   rckpt，agent 场景每回合省整段"上条回复"的重 prefill——用户感知可能超过所有 kernel 项。
   - 前置：无（aca5987 之后未动）；要实现 rckpt 数量上限与淘汰、MTP 成批接受后的对应位置
     存点、并发 slot 显存预算。
   - 验证：`tools/tcache_verify.sh` 加 agent 回放场景（两轮工具调用，第二轮 `cached` 须覆盖到
     `<tool_call>` 处）+ `api_regression_test` 90/90（§9 D1/D2 口径）。
   - 收益：agent 多轮场景每回合省一次整段重 prefill（几千 token 量级）。

2. **小 P：gemv_multi P 上限上推**（结论出自 §3 A3 状态段）：小 P 对 gufo/halogen 差 20%+；
   M1 实测 v2 ht_deq 占比 P=256/512/1024/2048 = 19.5/13.3/10.7/4.4%。trellis 解码进 WMMA A
   加载器太复杂（`k_ht_gemv` 只适合 gemv 布局），现实路径是把 `gemv_multi` 的 P 上限上推
   （P≤32 时 gemv 直读仍划算，只调常数）。
   - 前置：无（A3 的 w4b q4cp LUT-dense 原型已判定不做，harness 留存在 tools/moe_lut_test.cu
     `dense_mode` + logs/a3_dense_lut.md）。
   - 验证：M1 脚本小 P 两档（P=512/2048，§9 pp 口径）；gemv 与 WMMA 路径结果不同 → 非逐 bit，
     过 c512 KLD 门槛（同机改前值 +0.002）。
   - 收益：P≤1024 时 ht_deq 占 10.7–19.5%，上推 gemv 覆盖区间可砍掉其中大头；16K 稳态 ≈0。

3. **C2 gdn:scan 深流水**（§5 C2）：`k_gdn_fused` 受延迟与占用率限制（48 CTA/20 WGP=3 波，
   波效率 ~60%），LDS 已 57/64 KB 只能靠寄存器倒手；GDN 维度唯一没被否决的形态。
   - 前置：无；正确性 harness 复用 `tools/gdn_fused_proto.cu`。
   - 验证：proto 逐 bit/误差门槛 + ktest gdn 系列 + c8192 KLD（`--maxctx 8192`）；pp 128K 背靠背。
   - 收益：上限 ~3.5%（128K 口径），收益不确定；工作量数天。

4. **机器降级态排查**（§9 已记判定）：10-02 → 10-03 跨日 pp 1443.7 → 1228.7 tok/s（v2
   128k@16384），系统安静下空闲探针确认**非** CPU 抢功耗（但并行 grep 污染确实存在，测量纪律
   已立 §9）。方向：硬件/驱动/热，用户侧排查。
   - 前置：无（引擎侧数据已齐：idle_probe_v2.log 1228.7 vs 10-02 的 1443.7）。
   - 验证/口径：跨日绝对值比较前必须先跑探针校准（同一命令：
     `PREFILL_CHUNK=16384 bash tools/pp_win.sh 128k <tag>`）。
   - 收益：若恢复，所有 128K pp 绝对值 +17% 量级（相对降级态）。

5. **小项两条**：
   - **select f16 向量化**（§5 C1 状态段遗留）：select f16 目前 x0.87/x0.83 反而慢（2B 散读未
     向量化），改 uint4 一次 8 个 f16 + 对齐 peel。预估 ~0.2 ms/batch（128K 净账 1.7→1.9 ms）。
     验证：选块重合率 + 设备/host select 逐行一致（C1 口径）+ microbench。
   - **w4b gdn 段 +3% 现象**（§4 B1 w4b 状态段）：B1-w4b f16 pairs 后 gdn 段 +147 ms/16K
     （4874 vs 4727），方向在两次独立测量中一致，疑为 moe 段省时后的频率/功耗重分配。净账
     已为正（segsum −1.4%），排查属锦上添花；方向：同 run 内 KPROF 段间时钟/功耗采样。

---

## 11. 复核（10-03 晚，基于 logs/m1_m2_report.md 二读）

本节在 §10 之后补充，**优先级以 §11.6 为准**。

结论有四条：

- M1 的数据里藏着一个当时被当成噪声的异常，它是目前短 prompt 场景最大的单项（F1）。
- M1 剔除了第 1 个 chunk，恰好漏掉了短 prompt 的真实场景，需要补测（M1b）。
- A3 的"不做"判定依据有误，应按 P 分流重新在引擎里 A/B。
- 在 P=2048 下，路由 MoE 已与 gufo 持平。剩下的差距分散在 dense/HC、注意力和 reduce，没有单一大头。

§8 与 §10 里其余的否决和结论不受影响：差距都在 3% 以上，降级态也推翻不了。

### 11.1 F1：前 2051 个位置的注意力改走 WMMA（新，短 prompt 头号项）

**现象。** M1 两套权重的 `qsa:flash` 都出现 P=1024 比 P=2048 还慢的反常。M1 报告写的是"选块/SSD 噪声，未追"，但它在两套权重上逐档复现：

| P | v2 qsa:flash | w4b qsa:flash |
|---|---|---|
| 512 | 81 ms | 79 ms |
| 1024 | 151 ms | 148 ms |
| 2048 | 108 ms | 106 ms |

**原因。** `qsa_flash_b`（40_model.inc ~2585）把每个 batch 拆成两段：

- **稠密前缀**：`dense_P = min(P, 2051 - base)` 行，走 `k_qsa_flash<false, false, KVT>`（22_kernels_prefill.inc:796）。
  - 这是 fp32 标量实现，每个 block 处理 16 行 q 乘 1 个 q head，grid 为 (dense_P/16, 24)。
  - GQA 是 12:1，同一份 K/V 被 12 个 q head 各读一遍，也不用 WMMA。
- **稀疏尾段**：绝对位置 ≥2051 的行，走 `k_qsa_wmma`。一个 block 负责一个 (token, kvh)，12 个 q head 拼成 WMMA 的 M 维，K/V 只读一次。

**代价模型。** 按"每对 (q token, key)，12 个 QSA 层、24 个 head 合计"折算 w4b 的 M1 数据：

| 路径 | 依据 | 每对开销 |
|---|---|---|
| 稠密 | P=512：第 2、3 个 chunk 的位置 512–1535 全在稠密段，平均 524K 对 / 79 ms | ~151 ns |
| 稀疏 WMMA | P=2048：4.2M 对 / 106 ms | ~25 ns |

稠密路径大约慢 6 倍。用这两个系数回推另外两档：

- P=1024：第 2 个 chunk 全稠密，238 ms；第 3 个 chunk 基本是稀疏，53 ms；平均 145 ms，实测 148 ms。
- P=256：20 ms，实测 21 ms。

模型自洽。

**影响。** 每个对话的前 2051 个 token 共 2.1M 对，按标量路径要付 ~318 ms，改走 WMMA 后约 53 ms。下表把 prompt 视为从 base 0 起的单个 chunk，墙钟按 M1 稳态换算，属于估算：

| prompt 长度 | 现在（估） | F1 后（估） | 收益 |
|---|---|---|---|
| 1K | ~946 ms | ~880 ms | ~7% |
| 2K | ~1840 ms（≈1110 tok/s） | ~1575 ms（≈1300 tok/s） | ~17% |
| 3K | ~2490 ms | ~2225 ms | ~12% |
| 8K | ~5.9 s | ~5.6 s | ~4.5% |

几点补充：

- 对 gufo 的 pp2048@d0（1628）而言，我们真实的 d0 只有 ~1110 tok/s，差距约 32%，比 M1 表里的 21% 更大，因为 M1 剔除了这一段。F1 能收回其中约一半。
- 多轮增量（base > 2051）不受影响。
- 16K chunk 稳态只受第一个 chunk 影响，摊到长 prompt 后不足 1%。

**做法（推荐：给 `k_qsa_wmma` 加一个 `Dense` 模板参数，不新写 kernel）。**

kernel 内只有三处要改：

- `ntok = token + 1`，原来是 `2048 + (token+1)%4`。
- 源位置恒等：`src = gkey`。TransposedV 分支取 `vs = vk`，`vk` 已按 4 对齐。不再读 `blocks[]` / `tail0`。
- 迭代数改为 `(ntok + QW_KITER - 1) / QW_KITER`。原来固定为 `QW_NITER = 17`，稠密段一定 ≤17，循环上界改成运行时值即可。注意 `#pragma unroll 1` 已存在，不会多出寄存器。

调用侧在 `qsa_flash_b` 里改：

- `dense_P > 0` 且 KVT 为 bf16、`qsa_wmma` 开时，launch `k_qsa_wmma<TV, Paged, /*Dense=*/true>`，`grid = (dense_P, 2)`，`first = base`。
- 其余配置（fp32 KV、`GDEC_QSA_DENSE`、`qsa_global_v`）保持原路径。
- 要覆盖全部实例化：`TransposedV` true/false × `Paged` true/false。
- 调用点：3142 与 3214（主干），以及 4817 一带的 MTP 层 bf16 分支（`d_mkcb`）；fp32 分支 `d_mkcf` 不动。

为什么可以复用：

- 稠密前缀与稀疏尾段的 Q 布局、KV 布局、softmax 和输出写法完全相同。区别只在"看哪些 key"。
- 4-slot 组尾的问题已由 `k_f32_to_bf16_v4_bt` 把 batch 尾部写成 0 解决（22_kernels_prefill.inc:185 注释）。
- 组内大于 token 的 slot 是本 chunk 内的有效行，由 `gkey >= ntok` 屏蔽。这一点与稀疏 tail 现有行为相同，chunk 切分相关的 LSB 性质也一样，不引入新问题。

预期 kernel 性能：

- 稠密段的 key 是连续的，没有 `blocks[]` 间接寻址。每对开销应 ≤ 稀疏的 25 ns。
- 小 token（t < 128）的 block 只跑 1 次迭代，Q staging 的固定开销占比大，但总量很小（≤128×2 个 block）。

**验证。**

1. **microbench / 设备端对拍**：在稠密段对照旧 fp32 `k_qsa_flash<false>` 的输出。误差口径沿用 k_qsa_wmma 注释里的"maxabs ~1.6e-4 vs bf16-P fp32 参考"，门槛按同量级设。另加 P=1、P=17、base 非 0 且 base+P 跨 2051 的边界用例。
2. **c512 KLD**：64×512 的上下文整段落在稠密段，所以这正是测 F1 最准的口径。门槛：v2 0.10699+0.002；w4b 以同机改前值 +0.002。需在机器恢复后跑，或背靠背跑。
3. **needle**：用 YARN-512K-RESULTS 的方法跑短档（needle 位于 <2K 处）。
4. **pp**：用 M1b（§11.2）测 1K/2K/3K 的 TTFT。

**风险。**

- 非逐 bit：fp32 改为 bf16 Q/P 加 WMMA 累加。不过 ≥2051 的行早已是这种数值，质量先例在。
- kvsnap/rckpt 里旧二进制存的快照与新数值会有微小差异，不影响正确性。如果指纹含数值版本，按惯例 bump。
- decode（k_qsa_flash Split）不变。

工作量：半天到一天（模板改动 ~30 行，加对拍和验证）。

**状态（10-03 晚，已实现并默认开，验证过线）**：`k_qsa_wmma` 加第三模板参
`bool Dense=false`（22_kernels_prefill.inc），kernel 内三处按规格改动：`ntok=token+1`、
QK 与两个 V 分支的源位置恒等（`src=gkey`/`vs=vk`，Paged 的 `kv_row`/`ik_blk` 保留）、
迭代数改运行时上界 `(ntok+QW_KITER-1)/QW_KITER`，`blocks`/`tail0` 在 Dense 下不读。
调用侧 `qsa_flash_b`（40_model.inc:2780）：`dense_P>0 && bf16 KV && qsa_wmma &&
!qsa_dense` 时 launch `<TransposedV,Paged,Dense>` 四实例化（`grid=(dense_P,2)`，
`first=kbase=base`，`selected=nullptr`）；fp32 KV / `GDEC_QSA_DENSE` 保持原 fp32 标量
路径。MTP bf16 分支（`d_mkcb`）共用 `qsa_flash_b`，自动覆盖。**默认开、无独立
opt-out**（回退 = `GDEC_QSA_WMMA=0` 全退 WMMA）。验证：

- 编译 + ktest ALL PASS（稀疏路径 `Dense=false` 默认，零变化）。
- **c512 KLD**（64×512 全落稠密段，含 base=2048 跨 2051 的 dense_P=3 边界）：v2
  **0.107066** ≤ 0.10699+0.002 ✓；w4b **0.163505** ≤ 同机改前 0.163530+0.002 ✓
  （logs/kld_f1_{v2,w4b}_c512.log）。
- 边界 smoke：P=1、P=17 均 rc=0、decode 出正常 id、无 error/NaN（logs/f1_edge_p{1,17}.log）。
- **M1b TTFT 前后**（v2，健康机器，PLE 暖、中位数；qsa:flash 为 KPROF 段值）：

  | P（实际 tokens） | 前 tok/s | 后 tok/s | Δ | qsa:flash 前→后 |
  |---|---|---|---|---|
  | 512（~640） | 760 | **791** | **+4.1%** | 33→6 ms |
  | 1024（~1220） | 986 | **1086** | **+10.1%** | 112→18 ms |
  | 2048（~2350） | 1066 | **1212** | **+13.7%** | 339→64 ms |
  | 3000（~3350） | 1161 | **1273** | **+9.6%** | 388→114 ms |
  | 8192（~9170） | 1283 | **1329** | **+3.6%** | 709→440 ms |

  稠密段 qsa:flash **−81%**（2K 档 339→64 ms，每对开销 151→~29 ns，代价模型兑现）。
  与上文估算（+7/+17/+12/+4.5%）同量级略低（实际 tokens 偏多、其它段占比）。
  **对 gufo pp2048@d0 的差距 34.5% → 25.6%**（1212 vs 1628），收回约一半中的大头。
- **16K 稳态不退**：128k@16384 抽跑 1452.7 vs 1457.7（−0.3% 噪声）；chunk 1
  1304→1333、warmup dummy 1420→1456（首 chunk/dummy 含稠密前缀，直接受益）。

遗留：ktest 级 maxabs 设备对拍（vs fp32 稠密参考，本轮以引擎 KLD + 边界 smoke 代替）、
needle 短档（<2K）、独立 opt-out env（如 `GDEC_QSA_DENSE_WMMA=0`）。

### 11.2 M1b：首 chunk / 单 chunk 补测口径

M1 取第 2、3 个 chunk 是为了避开 PLE 冷读，但代价是测不到下面两种情况：

- 生产 `PREFILL_CHUNK=16384` 时，**≤16K 的 prompt 就是一个 P=prompt 长度、base=0 的 chunk**。
- 多轮增量是一个 P=新增 token 数、base=已有上下文的 chunk。

前者的大头恰好是 F1 的稠密段。补测口径：

- **单 chunk TTFT**：
  - prompt 长度 ∈ {512, 1024, 2048, 3000, 8192}，`PREFILL_CHUNK=16384`。
  - 同一进程里先发一个 64 token 的 dummy 请求暖热 PLE，再发正式请求，只计正式请求的 prefill 墙钟。
  - 每档 3 次取中位数。
  - 同时开 KPROF，报 qsa:flash 和 dense 各段。
- **增量**：base≈32K 时续 P ∈ {256, 1024, 4096}，走 API 两次请求（§2 M1 第 1 条的做法），看 A3 和 dense 段。
- 脚本可以在 `tools/pp_psweep_win.sh` 上加 `MODE=ttft`。结尾打印表格，并与 gufo pp2048@d0 1628 / d4K 1523 并列。

M1b 先于 F1 跑一遍作为基线，F1 之后再跑一遍。这一档数字比 16K 稳态更接近用户体感。

**状态（10-03 晚，基线已测，F1 后复测见 §11.1 状态段）**：`tools/pp_psweep_win.sh`
加了 `MODE=ttft`——`--serve` + gdec-api 同进程多请求：每档每 rep 先发 64 token
dummy 暖 PLE（一次性表加载，不计时），再发正式 prompt（yarn_ladder filler，rung
唯一化防 TokenCache 前缀命中，`max_tokens=1`）只计正式请求墙钟，3 次取中位数；
引擎开 KPROF，按 kprof 行的 `P=` 匹配正式请求报分段。PREFILL_CHUNK=16384，
`--maxctx 17408`。用法：`MODE=ttft MODEL_FILE=…v2.hgn NGRAM_FILE=…ngram.hgn bash
tools/pp_psweep_win.sh`。F1 前基线（v2，健康机器）：

| P | 实际 tokens | TTFT ms | tok/s | qsa:flash ms |
|---|---|---|---|---|
| 512 | 633 | 833 | 760 | 33 |
| 1024 | 1201 | 1218 | 986 | 112 |
| 2048 | 2321 | 2178 | 1066 | 339 |
| 3000 | 3301 | 2842 | 1161 | 388 |
| 8192 | 9041 | 7049 | 1283 | 709 |

对 gufo pp2048@d0=1628 的差距 34.5%（§11.1 估 32%，吻合）。遗留：增量档（base≈32K
两次 API 请求）、w4b 基线。

### 11.3 A3 重新评估：按 P 分流在引擎里 A/B，不再用"生产 16K 稳态"否决

§3 A3 状态段的判定依据有两条：

1. 生产 chunk 16384 时 0.78–0.96× 稳输；
2. P<1024 时引擎实际走 hipBLASLt，按 Lt 口径只是打平。

第 1 条混淆了 chunk 上限和实际 P。如 §11.2 所述，16K chunk 下所有 ≤16K 的 prompt 和所有多轮增量都以小 P 运行。分流后 16K 稳态仍走原路径，零影响，所以它不构成否决理由。

第 2 条只对 P<1024 成立。**P∈[1024, 交叉点) 时引擎走的正是 `k_gemm_wmma`**（40_model.inc:2013 `P >= 1024`），ratio 表的对照对象就是引擎实际路径：

| 形状 | P=1024 | P=2048 | 交叉点 |
|---|---|---|---|
| d3 (2560,6144) | 1.25 | 1.13 | ~4000 |
| d9 (10240,2560) | 1.11 | — | ~1500 |

这一窗口里 LUT 是实打实地赢。另外，表里的 deq 是 L2 热的下界；引擎里每个 chunk 都要冷读一遍，只会更偏向 LUT。

做法：

- 加一个 `GDEC_DENSE_LUT_MAXP` 开关（d3 / d9 分开，默认值取交叉点的 80% 左右）。仅限 q4cp（dtype 5）dense。
- `gemm()` 里在 `gemv_multi_q4cp` 之后、`gemm_wbf16` 之前分流到 `moe_lut_down` 单专家，复用 `tools/moe_lut_test.cu dense_mode` 的调用方式。
- P<1024 先只做 A/B，不默认开：Lt 口径打平，要以 M1b 的引擎墙钟为准。
- 其余形状（N=12288/6144、HC 的 q8 系）先用同一 harness 补测 ratio，再决定是否纳入。

验证：

- 非逐 bit（f16 激活 vs bf16），过 c512 KLD 门槛（同机改前值 +0.002）。
- pp 看 M1b 的 1K/2K/3K TTFT 与增量 P=1024。

预期：w4b 在 P=1024–2048 时 dense 段快 10–20%，整体 3–6%；P=512 待 A/B。v2 的 ht 不适用，仍按 §10 第 2 条推 `gemv_multi` 的 P 上限。

### 11.4 P=2048 稳态下与 gufo 的逐段对照（修正此前"MoE 是追 gufo 的大头"）

口径：w4b，M1-w4b 的 P=2048 一档（第 2、3 个 chunk，深度 2–6K），单位 µs/token；gufo 取 GUFO-GAP.md 的 d0 profile，注意力按 d4K 略上调。

| 段 | 我们 | gufo | 差 |
|---|---|---|---|
| 路由 MoE（up+down） | 237 | 233（整段，含 router/reduce） | ≈ |
| MoE reduce + route | 33 | 0（gufo 把带权求和并进 HC combine） | +33 |
| dense + HC + shared（含 q4cp 反量化） | ~377 | ~310 | +67 |
| 注意力 + indexer | 74 | ~37–45 | +30 |
| GDN 核心（scan/conv/norm） | 73 | 63 | +10 |
| 合计 | 796 | ~657 | ~140 |

解读：

- 在 P=2048 下，我们的 MoE up+down 约 20 TF，与 gufo 的 MoE（~20 TF）持平。09-25 那次占差距 78% 的 MoE 问题已被 LUT kernel 解决。
- 剩余差距分散，没有单一大头。dense/HC 这一项 A3 能吃掉一部分；reduce 并入 HC 已由 B2-1 以 −5% 否决，不重提。
- P ≤ 1024 时，MoE 已在权重带宽的物理下限（M1 报告的 A2 判定成立），小 P 只剩 dense 和 F1 可压。

### 11.5 MoE kernel 结构项：只对大 P 有效，排后

`k_moe_lut` 的一个 CTA 负责 128 行权重乘一个 ≤64 行的 token tile（`MoeTile`，BN≤64 由 static_assert 限定）。同一个专家的多个 tile 由不同 CTA 处理，各自把同一批权重重新解码一遍。A1 的结果（加 VALU 就退步）说明解码 VALU 在关键路径上。因此：

- **只在单个专家行数超过 64 时才有可摊薄的重复解码**，也就是平均每专家 >64 行，约 P ≥ 3.3K。16K 时平均 320 行（5 个 tile），重复解码 5 次。
- P ≤ 2048 时平均只有 ≤40 行，没有重复。此前估计的"2048 下 ~8%"不成立，这里更正为 ≈0。
- 方向：一个 CTA 连续处理同一专家的 2–4 个 tile。把解码后的 f16 A 片段写进 LDS（128 行 × 32 K × BK2 ≈ 16 KB），多个 token tile 从 LDS 读 A。A1 已证明 LDS 读能藏在 WMMA 之后，这样就把 VALU 解码换成了 LDS 读。
- 不要再走"A 片段留在寄存器"那条路：BK=4 和 2 个 A 片段都已因 VGPR 否决。
- 预期：16K 时 up+down 2889 ms 中解码部分减半，整体 ~4–6%。这只是推算，先在 `moe_lut_test` 里做原型，数天工作量。
- 适用于 128K/16K 长 prompt。短 prompt 和多轮增量都不受益。

### 11.6 修订后的优先级（覆盖 §10 的顺序，§10 各条内容仍有效）

| 序 | 项 | 场景 | 预期 | 工作量 | 前置 |
|---|---|---|---|---|---|
| 0 | 机器降级排查（§10-4）+ 整栈"新默认全开 vs 全关"复测 | 全部 | 绝对值 +17% 量级；确认 10-03 各项收益 | 用户侧 | — |

序 0 已闭环（10-03 晚）：机器重启后恢复健康（128k 探针 1469.7、256K YaRN 1397.0；
老二进制 v0.0.5 复跑 1126.5 vs 其健康基线 1443.7 = −22%，确认真降级非代码）。健康
机器全开 vs 全关（v2 128k@16384 同链相邻）= **1457.7 vs 1443.1（+1.0%）**；降级态口径
曾 +6%，机制：健康态不再带宽受限，f16 pairs / index f16 的绝对收益缩小。
| 1 | M1b 基线（§11.2） | 短 prompt / 增量 | 无直接收益，给 F1/A3 定基线 | 1–2 小时 | — |
| 2 | **F1 稠密前缀 WMMA**（§11.1） | 所有对话的前 2K token | 1K +7%、2K +17%、3K +12% | 半天–1 天 | M1b |
| 3 | **A3 按 P 分流**（§11.3） | w4b / GGUF，P 1K–4K | 整体 3–6% | 1 天 | M1b |
| 4 | D1 rckpt（§10-1） | agent 多轮 | 每回合省整段重 prefill | 中 | — |
| 5 | v2 `gemv_multi` P 上限（§10-2） | v2 小 P | ht_deq 10–20% 中的大头 | 小 | M1b |
| 6 | MoE 多 tile 共享解码（§11.5） | P ≥ 4K、长 prompt | 16K ~4–6% | 数天 | 机器恢复 |
| 7 | C2、select f16 向量化、w4b gdn +3%（§10-3、§10-5） | 长上下文 | ≤3.5% / 0.2 ms / — | — | — |

测量纪律补充：

- F1 与 A3 的验收以 M1b 的 TTFT 为主，16K 稳态为辅（只需确认不退）。
- 机器恢复前，只做背靠背的相对比较，不跨日比较绝对值（§9）。
