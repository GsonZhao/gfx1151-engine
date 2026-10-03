# v2 权重 prefill 优化方向（2026-10-02，交接给 Windows 实机）

背景：11b6847 之后 Linux 上 v2 prefill 已与 w4b 持平（pp32K@16384 1519 vs 1512 tok/s，
128K 1444 vs 1446）。本文列出剩下的方向，按"收益/工作量"排序。**所有估算来自 10-01 在
Linux 上的 microbench（/tmp/htb，16K chunk），Windows 上没测过**；Windows 整体比 Linux 慢
~25%（PORTING-WINDOWS.md：32K@16384 1068 vs ~1422），原因一直没定位，所以先做方向 0 再排后面。

时间基准：Linux 16K chunk ≈ 10.8 s（16384/1519）。下文"ms/16K"都是每个 16K chunk、48 层合计。

| # | 方向 | 预期 | 工作量 | 风险 | 状态（10-02 Windows 实机） |
|---|------|------|--------|------|------|
| 0 | Windows 上可用的分段计时（GDEC_KPROF） | 无直接收益，决定后面怎么排 | 半天 | 无（默认关） | **已完成** |
| 1 | Windows + v2 用 PREFILL_CHUNK=16384 | +6%（w4b 实测值） | 改配置 | arena 余量 | **已完成**（16384 成推荐配置） |
| 2 | down 输出 pairs 改 f16 | 1.7–3.4%，省 0.8 GiB | 小 | KLD | 已实现，**默认关**（GDEC_V2_PAIRS_F16） |
| 3 | k_i4r_mid 融进 LUT up 尾声 | ~2% | 中 | 复杂度 | 已实现（双趟结构），实测≈持平，保留 |
| 4 | k_moe_lut 本身提效（v2 和 w4b 都受益） | MoE −20~25%，整体 5–6% | 大 | 需原型 | ③+④ 已进引擎（kernel −11.7%）；②① 已否决 |
| 5 | rot_in 融进 norm | ~0.5% | 小 | 低 | **已否决**（10-03 分析：d_xb 生产者 Epi=1 每 CTA 只 32 通道，FWHT 需 128；见 PREFILL-ALL §4 B5） |
| 6 | ht dense 每 chunk 反量化 | ≤0.6%，不建议做 | — | — | 不做 |

---

## 当前状态（2026-10-02，Windows 实机）

累计 pp（v2，chunk 16384，128k 稳态 tok/s）：

| 阶段 | 128k | 说明 |
|------|------|------|
| 基线（方向 0 之前，8192 chunk） | 1385 | |
| 方向 1 之后（16384 chunk） | 1423.7 | +2.8% |
| 当前（方向 3 + 方向 4 的 ③+④ 进引擎） | 1433.3 | 三次均值 1440.1/1422.7/1437.2 |
| KPROF 细分埋点进源码（默认关，见下节） | 1439.0 | 零开销复测，噪声内 |

32k 当前 1521.5（只有 1 个稳态 chunk，噪声大，仅供参考）。方向 3 的 moe:mid 分段已消失
（151→0.1 ms/16K），方向 4 的 ③+④ 使 k_moe_lut microbench −11.7%，但两项落到 16384-chunk
pp 上都只有零点几个百分点（在 run 间噪声 ±0.6% 内；8192 对照下合计 +2.2%，见下）。方向 2
默认关，方向 5 未做。详细数据见各节末尾的"状态"段。

**收益分解（8192 chunk 对照，10-03 补测，`logs/v2_128k_c8k_moeopt.log`）**：当前二进制回
8192 chunk 跑 128k，稳态 avg **1415.5**（min 1340.0 / max 1508.0，n=15，剔除预热+首块）。
累计 +3.9%（1385→1439.0）拆成两部分：

| 阶段 | 128k tok/s | 增量 | 性质 |
|------|-----------|------|------|
| 基线（8192 chunk，方向 0 之前） | 1385 | — | |
| + MoE 方向 3+4（同 8192 chunk） | 1415.5 | **+2.2%** | **真实算力收益**，任何 chunk 成立，bit-exact |
| + chunk 16384（方向 1） | 1439.0 | +1.7% | 显存换吞吐（devarena 78→85 GiB），力大砖飞 |

注意两点：① +2.2% 高于按 KPROF 分段折算的预估（~1.1%），疑因 8192 下 launch 次数翻倍使
方向 3 的 mid 融合收益翻倍；且 1385 与 1415.5 各为单跑，run 间噪声 ±0.6%，真实区间约
+1.6~2.8%。② **16384 配置仍保留为推荐**（显存够就用），但对外报收益时应区分「kernel
收益 +2.2%」与「chunk 收益 +1.7%」两部分，不要把 +3.9% 全算在优化头上。

重要结论：v2 在 Windows 已基本追平 Linux（16K chunk segsum 10.93 s vs Linux 10.8 s），
PORTING-WINDOWS.md 里"Windows 慢 25%"对 v2 不成立（那是 w4b + 8192 chunk 的数字）。

## 0. 先做：Windows 能用的分段计时

rocprofv3 / KTRACE 只在 Linux 上跑过。Windows 上现有的两个开关都不够用：
`GDEC_PHASE=2` 每层同步一次，只给出层总时间（看得出 QSA 层和 GDN 层差多少，但分不出
attention 和 MoE）；`GDEC_PROF=1` 只计 host 侧时间。

建议加 `GDEC_KPROF=1`（默认关，关时零开销）：

- 启动时预分配一个 hipEvent 环（例如 8192 个），宏 `KP("tag")` 在 `g_str` 上 `hipEventRecord`
  并记下 tag；**chunk 内不做同步**。
- chunk 结束（`40_model.inc` 里 `phase("argmax_d2h")` 之后，约 4401 行）同步最后一个 event，
  用相邻 event 的 `hipEventElapsedTime` 求差，按 tag 累加后打印一行汇总。
- 打点位置（都在 `40_model.inc`）：
  - `enq_layer_b`（3952）：`gr_read/gr_write`（HC）、`qsa_b` / `gdn_b`、`moe_b` 三段；
  - `moe_b_v2`（3812）LUT 分支内部：shared（`moe_b_shared_ey`）、routing+tiles、`v2_rot`、
    `moe_lut_up_raw`、`k_i4r_mid`、`moe_lut_down`、`k_v2_reduce_rot`；
  - `gemm_wbf16`（1883）里 `v2_deq_bf16` 前后单独记，看 ht 反量化在 Windows 上的真实占比。
- 注意：Windows 有 `prefetch`、`pipe_str`、`ovl_str` 等其他流。只在 `g_str` 上打点时，
  某段如果在等别的流，等待时间会算进这段。正好用来看 PLE 预取有没有拖后腿。

有了它要回答两件事：
1. Windows 比 Linux 慢的 25% 集中在哪几段。如果集中在少数 kernel，先查**编译器差异**：
   TheRock 的 hipcc 和 Linux ROCm 版本不同。在两边各编一次，加
   `-Rpass-analysis=kernel-resource-usage`，对比 `k_moe_lut<kI4R,…>`、`k_gemm_wmma`、
   GDN kernel 的 VGPR、spill 和 LDS。这一步很便宜，可能直接解释掉一大块。
2. 方向 2–5 在 Windows 上各自占多少，据此决定做不做。

**状态（已完成）**：GDEC_KPROF 已落地——`07_kprof.inc` + `KP`/`KP_SUB` 埋点，关时零开销，
图捕获期屏蔽；Windows 上 hipEvent 偶发回跳已钳负。（10-02 扩展：flush 支持多 sub tag +
嵌套栈配对，gdn/qsa 细分见文末「GDN/QSA 深挖」。）v2 稳态分段（剔除前 2 个 chunk，PLE 冷读
会污染对比），每 16K chunk、48 层合计：

| 分段 | ms/16K | 占比 |
|------|--------|------|
| gdn | 3137 | 28.7% |
| qsa | 2062 | 18.9% |
| moe:up | 1810 | 16.6% |
| moe:down | 1094 | 10.0% |
| hc 系列 | ~1670 | 15.3% |
| moe:reduce | 446 | 4.1% |
| moe:shared | 321 | 2.9% |
| moe:mid | 151 | 1.4% |
| moe:rot_in | 57 | 0.5% |
| ht_deq | 71.5 | 0.7%（含在上列分段内） |

segsum 10.93 s/16K，已基本追平 Linux 的 10.8 s（见文首"当前状态"）。

## 1. 配置：Windows + v2 用 chunk 16384

Windows 默认 chunk 8192（`30_host_util.inc` `eff_maxbatch`），原因是 w4b 在 256K 下用
16384 会超出 95 GiB 的 arena 上限。v2 权重小了约 6 GiB（RSS 62.9 vs 69.1），可能放得下。
w4b 在 Windows 实测 16384 比 8192 快 ~6%：32K 920→979，两次都是 maxctx 40K 下测的。
chunk 变大还会摊薄每 chunk 一次的 ht 反量化，并让 MoE tile 填得更满（16K 时每个专家
平均 320 行，8K 时 160 行，BN=64 的尾 tile 浪费更少）。

做法：在 service.conf 设 `PREFILL_CHUNK=16384`（`launch_win.cpp:909` 读它，>0 时才导出
`GDEC_PREFILL_CHUNK`）。`start_win.sh` 本身不读 PREFILL_CHUNK，用它启动时要直接
`GDEC_PREFILL_CHUNK=16384 bash start_win.sh`。

判断标准，看引擎日志：

- `devarena: requesting X GiB (estimate E + slack 4.00 GiB)` 这一行里 **E 必须 ≤ ~91**。
- 出现 `arena: estimate … exceeds cap 95.00 GiB; clamping` 只说明 slack 被压缩了，本身不算错误；
  真正的问题是后面又出现 `fall back to hipMalloc`，那表示有缓冲落到了 arena 外。
- 粗算：w4b 256K/8192 serve 实测估算 91.03 GiB，16384 再加 ~6.9 GiB，v2 减 ~6.2 GiB，
  所以 E ≈ 91.7。很紧，估算偏高还是偏低要以实机日志为准。做完方向 2 能再省 0.8 GiB。
- 放得下就跑一遍 32K 和 128K 的 pp，与 8192 对比；跑完再开一个长对话看会不会卡死。

可选的引擎改动：Windows 上如果 `devarena_estimate`（chunk 16384）+ slack ≤ cap，就自动选
16384。注意 `devarena_estimate` 内部调用了 `eff_maxbatch`，不能在 `eff_maxbatch` 里直接
调用它（会递归）。应该在 `devarena_init` 前先算一次，再写进一个类似 override 的全局变量。
建议先用配置验证收益，再决定要不要改引擎。

**状态（已完成）**：`pp_win.sh` 已改为读 `PREFILL_CHUNK` 环境变量（默认 8192）。16384 实测：
32k 1450→1498.3（+3.3%）、128k 1385→1423.7（+2.8%）；devarena 余量充足（128k@16K 估算
85.26 GiB，远低于 95 GiB 上限）。结论：**16384 成为 v2 on Windows ≤139264 maxctx 的推荐
配置**。256K serve 未实测（粗算 E≈91.7 很紧），上线前要实机验证。

## 2. down 输出 pairs 改 f16

现状：`moe_lut_down` 写 f32 pairs `[P·k][2560]` 到 `d_guvb`，每层 16K 就是
16384·10·2560·4 = 1.68 GB；`k_v2_reduce_rot<10>` 再读一遍。reduce 实测 434 ms/16K，
基本被带宽卡住。

改法：
- `27_kernels_moe_lut.inc` 尾部 down 分支（322–339 行，16×16 LDS 转置后的
  `out[...] = tile_scratch[flat]`）改为写 `__half`。最好两两打包成 `__half2`：
  当前每个 lane 写 1 个 f32，改 f16 后合并写需要调整转置布局，否则写事务变窄。
- `28_kernels_hgn_v2.inc` 的 `k_v2_reduce_rot`（170）改为读 half。
- `30_host_util.inc` `devarena_estimate` 里 guvb 那一项也要改，但要先确认 w4b 路径不再
  用 f32 pairs 才能减。

收益：reduce 读减半 ~−180 ms，down 写减半最多 ~−180 ms，合计 1.7–3.4%，`d_guvb` 少 0.84 GB。

风险：pairs 是旋转域的 z（还没乘 svh、没做 H）。f16 最大 65504，量级应该没问题，但精度
只有 11 bit。bf16 量程更大、精度更低，不建议用。以 KLD 为准（见文末验证）。早先
`GDEC_MOE_PAIRS_BF16` 在旧 kernel 上测过，速度持平（PREFILL.md §10），但那时 reduce
不是瓶颈，现在情况不同了。

**状态（已实现，默认关）**：f16 pairs 已做进引擎，但实省只有 ~1.3%——reduce −33% 符合预期，
down 写减半无收益（down 不是写带宽受限）；且 KLD 自 A/B 0.0204 偏大、相对门槛存疑，故改为
`GDEC_V2_PAIRS_F16=1` 显式开启、**默认关**。遗留：等 `data/kld/bf16_c512.kld` 从 Linux 拷
过来后用 0.10699+0.002 的门槛复测定夺；0.84 GiB arena 节省未做（w4b 与 v2 共享 `d_guvb`，
w4b 路径仍是 f32 pairs）。

## 3. k_i4r_mid 融进 up kernel 尾声

现状：`moe_lut_up_raw` 写 f16 gate|up（`guv16`），`k_i4r_mid`（`28_kernels_hgn_v2.inc:124`，
一个 block 处理一个 slot：FWHT → ·svh → silu(g)·u·suh_dn → FWHT）读回来，写 f16 `hid`。
mid 133 ms/16K，guv 写 ~90 ms/16K。

难点：FWHT 跨 128 行（mid 维按 128 一组旋转），而现在 up tile 的 BM=128 只有 64 gate 行加 64 up 行
（`kPair` 时 `kRows = BM/2`）。要融合，tile 必须是 **128 gate 行 + 128 up 行（BM=256）**。

方案：
- 新模板参数，例如 `kMid`。8 个 wave × 32 行，每个 wave 持有 2 个 A 片段。B 片段从 LDS 读一次
  喂两次 WMMA，这正好也是方向 4 的寄存器分块，两件事可以一起做。
- mid 方向 grid = 640/128 = 5。
- 尾声按 token tile（16）逐个处理：256 行 × 16 × 4 B = 16 KB LDS。依次做 gate/up 各自 FWHT
  → ·svh → silu·u·suh_dn → FWHT，写 f16 `hid`。FWHT 代码复用 `hv2::fwht128`。
- `moe_b_v2`（`40_model.inc:3842` 附近）去掉 `k_i4r_mid` 那一次 launch。

收益 ~220 ms/16K（~2%）。风险：VGPR 翻倍后要确认 occupancy 不下降（`-Rpass-analysis`），
其次是代码复杂度。检验办法：融合版与现版的 `hid` 应在 f16 舍入误差内一致。

**状态（已实现，实测≈持平但保留）**：本文原方案（BM=256、每 wave 2 个 A 片段）被方向 4 的
microbench 否决（该结构 VGPR 101→156、up 回退 +8.5%），实际实现改为 **kMid 双趟结构**：
`k_moe_lut` 加模板参 `kMid`（kLut==kI4R、!kPair），每块持 128 gate + 128 up 行（正好一个
FWHT group，grid.x = mid/128 = 5）；pass 0 跑 gate 行的 K 流水线、acc 以 f16 落 LDS plane
（与 guv16 写盘同样的舍入），pass 1 复用 stage 区跑 up 行落第二个 plane，随后每 wave 处理
一个 slot 做 fwht128→·svh→silu(g)·u·suh→fwht128 直接写 hid f16，op 顺序与 `k_i4r_mid`
逐字一致。`hv2::kRs128`/`hv2::fwht128` 从 28 移到 27（include 顺序）；新增 launcher
`moe_lut_up_mid`，`moe_b_v2` 里替换掉 `moe_lut_up_raw` + `k_i4r_mid` 两次 launch。

验证：microbench 设备端 A/B 逐 bit 相同（0/13107200 mismatch），引擎 KLD 自 A/B bit-exact；
KPROF 里 moe:mid 段消失（151→0.1 ms/16K），up+mid 合计净省 ~225 ms/16K。但双趟重读 x 多走
~840 MB DRAM，正好抵消省下的 guv 往返（420 MB 写 + 420 MB 读），pp 净收益≈0（1433.3 vs
1429.5，在噪声内）。（10-03 补：16384 chunk 下摊薄到噪声内≠没有收益——8192 chunk 对照
下方向 3+4 合计 +2.2%，launch 次数翻倍使融合收益显现，见文首「收益分解」。）保留原因：
非负、零数学风险、省一次 launch、流量结构更好。遗留：kMid
VGPR=169 → occupancy 1 CTA/CU，spill 写 plane 有 16-way bank conflict，是将来追收益的
入手点。

## 4. k_moe_lut 本身提效（MoE 主体）

现状：w4b LUT 的 up 约 27.9 TF、down 约 23 TF，只有实测 WMMA 峰值 55.4 TF 的一半左右
（dense `k_gemm_wmma` 能到 37.9）。MoE 约占 chunk 时间的 26%。v2 和 w4b 共用这个 kernel，
做成了两边都受益。

`27_kernels_moe_lut.inc` `k_moe_lut`（72）里看到的问题：
- **B 片段复用低**：每个 wave 只有 16 行（1 个 A 片段），每个 token tile 的 B 片段（4×uint4）
  每个 wave 都从 LDS 读一遍（257 行）。改成每 wave 32 行、2 个 A 片段，LDS 读取减半。
- **barrier 太密**：BK=2，即每 64 个 K 两次 `__syncthreads`（271 行起）。down 的 K 只有
  640，共 10 个 stage，序言、尾声和 barrier 占比更高。可以试 BK=4，配合双缓冲 LDS。
- **解码走 LDS 查表**：每 32 个权重 16 次 `s_cbp[byte]` 查表加 `__hmul2`（约 248 行）。i4r 的
  码本是线性的 (nib−8)，可以改成算术解码：0x6400 magic 加 `v_perm` / and-or 直接拼出 f16，
  把 −8 并进 fma 的偏置，不再读 LDS。q4cp 的码本不是线性的，仍需查表，所以这条只对 v2 有效。
- **i4r scale**：每个 kb 都单独 load 一次 scale（`f_sp + (kb>>2)*2`，174 行），而它每 4 个 kb
  才变一次。可以像 q4cp 那样整块读进寄存器。

做法：先做 microbench 原型，不要直接改引擎。`tools/moe_lut_test.cu` 已能对真实 q4cp 专家做
正确性检查（CPU double 参考）和计时，但目前不支持 i4r。给它加 `--v2 X.hgn` 读 dtype 23 专家
（CPU 参考按 HGN-V2.md 的 (nib−8)·scale 解码），再拿新旧 kernel A/B。原型能快 ≥15% 再进引擎。
收益上限粗估：MoE 时间 −20~25%，整体 5–6%。这是剩下最大的一块，也最费工夫。

已否决：int8 WMMA。RDNA3 上 iu8 和 f16 吞吐相同，没有收益。iu4 快 2 倍，但激活也要 4 bit，
质量不可接受。

**状态（③+④ 已进引擎，②① 否决）**：`tools/moe_lut_test.cu` 已加 `--v2`/`--opt N` 支持
i4r 原型（dtype 23 专家，CPU double 参考 + 计时），四项 microbench 实测：

| 项 | 实测 | 结论 |
|----|------|------|
| ③ 算术解码 | −10.9%，bit-exact | 进引擎 |
| ④ scale 缓存 | −1.4% | 进引擎 |
| ② BK=4 | +9.7% 回退（VGPR 169） | 否决（occupancy） |
| ① 双行复用 | +2.9% 回退（VGPR 156） | 否决（occupancy） |

③+④ 组合 −11.7%，低于原定的 15% 进引擎门槛，但因 bit-exact 零风险，拍板进引擎（已完成）：
`27_kernels_moe_lut.inc` 的 kI4R 分支用 `0x6400+nib` 算术解码 + `__builtin_amdgcn_perm`
拼半字（字节池 {b:0-3, a:4-7}）替代 s_cbp 查表；scale 每 2 stage 读一次，s_scale/s_cbp
缩表；q4cp/IQ 分支逐 bit 未动（`if constexpr` 隔离）。引擎 KLD 自 A/B 逐 bit 相同，
microbench up 39.3→34.9 ms、down 22.0→19.3 ms，w4b 回归 PASS。②① 确认不可行（occupancy），
不必再试。上文"收益上限 5–6%、剩下最大的一块"是开工前估算；实测 pp 贡献含在 8192 对照的
+2.2% 里（见文首「收益分解」），**后续最大剩余项以「GDN/QSA 深挖」的剩余大鱼为准**。

## 5. rot_in 融进 norm

`moe_b_v2` 开头用 `v2_rot(d_xb → x16)` 做 suh·H/√128 并转 f16，约 56 ms/16K。router gemm
还要读未旋转的 `d_xb`，所以只能让写 `d_xb` 的那个 norm kernel（`gr_write_read_b` 那一路）
多输出一份旋转后的 f16。收益约 0.5%，改动小，但碰的是共享路径，需加 v2 判断。优先级低。

**状态：已否决（10-03）**。`k_v2_rot` 已在带宽顶（~212 GB/s），只能生产者融合；但现行配置
P≥1024 时 d_xb 由 `k_gemm_wmma` Epi=1 epilogue 写出，W 行置换后每 CTA 只有 32 个输出通道，
而 rot_in 的 FWHT 需要完整 128 通道组（跨 4 CTA），不重构 tile/置换做不了。小 P 路径
`k_gr_combine_b_hc_bf16`（一 token 一 block）本可行，但 P≥1024 不走它；为它关 hc_up_fused
净亏（HC 融合约值 250 ms/16K）。详细见 HANDOFF-PREFILL-ALL.md §4 B5。

## 6. 不建议：ht dense 缓存 bf16

`gemm_wbf16`（`40_model.inc:1883`）对 dtype 16/24 每次调用都反量化到 `d_wbf16`。实测多出
~30 ms/16K（≤0.3%，8K chunk 约 0.6%）。要消掉只能常驻 bf16 主干，会多占好几 GiB，
和 v2 省内存的初衷相反。

## 超出原文档的观察

方向 0 的分段数据出来后有一个原文档没覆盖的点：**GDN 28.7% + QSA 18.9% 合计近一半 chunk
时间，比 MoE 任何一个分段都大**，但不在任何方向里。如果还要继续追 prefill，这两块值得单独
评估（第一步很便宜：对照 Linux 编译同一份代码，比 VGPR/spill/LDS，先排除编译器差异）。
（10-02 已评估完毕，见下节「GDN/QSA 深挖」：与 Linux 差 ~5%，无编译器级差异；细分、否决
清单与剩余大鱼都在那节。）

## GDN/QSA 深挖（10-02，结论：无 ≥1% 数学不变方案，停在报告）

KPROF 细分埋点已进源码（默认关、关时零开销）：`07_kprof.inc` 的 flush 支持多 sub tag
+ 嵌套栈配对；`40_model.inc` 里 gdn_b 分 proj/conv/scan/norm/oproj 五段、qsa_b 分
idx/proj/prep/flash/gate/oproj 六段。子段和与外层完全对上（<0.1%），无隐藏时间。

### 细分耗时（每 16K chunk、48 层合计 ms，剔除前 2 chunk）

32k 稳态（segsum ≈ 10.65 s）：

| 段 | ms | 定性 |
|---|---|---|
| gdn:proj | 1423 | 92% 实测峰值（35.0/37.9 TF），4 个 in_proj gemm |
| qsa:flash | 940 | DRAM 流率极限（选中块 KV 读，L2 已吸收 ~3/4 邻近重叠） |
| gdn:scan | 695 | k_gdn_fused，延迟/占用率受限（48 CTA/40 CU，波效率 60%） |
| gdn:oproj | 646 | k_gemm_wmma k1 (2560,6144)，见「oproj 漂移」 |
| qsa:idx | 381 | t64 打分 8.7 TF，但非占用率受限（见否决清单） |
| qsa:proj | 378 | 94% 峰值 |
| gdn:conv / gdn:norm | 230 / 184 | 94% / 86% 带宽极限 |
| qsa:oproj / prep / gate | 209 / 84 / 84 | 峰值或带宽极限 |

128k 稳态（segsum ≈ 11.75 s）：gdn ≈ 3314 基本持平；qsa ≈ 2985，**随 base 涨的只有两处**：
idx 381→1032（打分 O(base)）与 flash 940→1371（KV 散布 + L2 命中率下降）。

### Linux 对比

HANDOFF-PREFILL-NEXT.md 09-28 trace（8K chunk×2 换算）：gdn_fused 656 vs 我们 695；
qsa_wmma 886 vs 940——**差 ~5%，无编译器级差异**，方向 0 的「Windows 慢」对 GDN/QSA
不成立。

### 实测否决清单（全部本机 microbench/原型实测）

- **fused3 波量化修复**（k_gdn_fused grid 48→96 半 chunk 流水，`tools/gdn_fused_proto.cu`，
  与串行 wgrad 参考逐 bit 一致）：长稳态（2000 token）与原 fused 打平（10.7% vs 10.6%），
  短测看到的 30% 是冷启动假象。**此路已死**。
- **t64 短 staging**（ST=8/4，LDS 25→12.5/6.3KB，2→4 CTA/CU，`tools/idx_s8.cu`，PASS
  bitwise）：只 +4~5%——t64 瓶颈在访存依赖链而非占用率。占用率修法无效。
- **gemm_lt 全扫**（12 配置 × gm∈{1,2,4,8,16}）：d9/gm=4 与 k1/gm=1 已是最优，无配置空间。
- **select_2p**：已在 ~220 GB/s 流率上，无占用率余量。
- **fused FEAT=8**（v 走 nontemporal load）：kernel 级 +4~6%、逐 bit 相同，但整体仅 ~0.3%。
- idx hist 融进打分 kernel 省一遍 select 读：128k 仅 ~1.3%、32k ~0.3%，全局原子量估算
  误差 ±2×，可能净负，不值。

### gdn:oproj 128k +14% 漂移：定名为物理内存放置运气（环境，不修）

- **现象**：gdn:oproj 稳态 646→738 ms/16K（+14%）、qsa:oproj 209→240（+15%），两者同为
  k_gemm_wmma k1 (2560,6144) 形状；同跑的 d9 形状 gemm（gdn:proj/qsa:proj）、moe:up、
  所有 elementwise 均平（≤+2%）。
- **排除热漂移**：128k 逐 chunk 数据从 chunk 1（base=0）起就恒高 733-762，非单调爬升；
  同 base=16384 的 chunk 在两跑中其余分段完全相同（idx/flash 逐位一致），功耗剖面无差异。
- **排除测量假象**：子段和与外层对上；无埋点的旧二进制同样 734 vs 646。
- **坐实布局**：32k tokens + `--maxctx 139264`（128k 布局跑短上下文）完整复现 +14%
  （728.5 vs 646）；`GDEC_GEMM_DBG=1` 指针打印显示两布局间 gemm 操作数地址仅 GB 级
  高位不同（低 28 位相同）。arena 是 256B 对齐 bump 分配，KV 随 maxctx 变大把批缓冲
  整体抬高（`30_host_util.inc` / `40_model.inc` KV 在批缓冲之前分配）。
- **pad 扫参**（新增 `GDEC_PAD_BATCH` knob，批缓冲前插入 dummy 分配，默认 0）：128k
  布局下 pad=2MB 时 oproj 回落到 656（≈32k 布局水平），但响应非单调、无周期
  （0.5/1/3MB→669-685，4/6MB→717-728），且**同一 2MB pad 会把 32k 布局从 646 打破到
  702**——不存在通用幸运偏移（扫参脚本 `tools/pad_sweep.sh` 留存）。k1 形状大量依赖
  L2 重用（A 被 20 个 N-tile 复读、B 被 256 个 P-tile 复读），对 L2 物理放置冲突敏感说得通。
- **判定不修**：收益上限 ~0.9%（仅 128k），低于 1% 门槛，且本质是重新抽签（换 maxctx/
  重启后失效）。`GDEC_PAD_BATCH` 留作诊断 knob。
- 注意：`GDEC_GEMM_DBG=1` 的 per-call fprintf 在 Windows 上极慢，会把 gdn:oproj 段
  污染 +35~100 ms/16K——只看它打印的形状/指针，别看时间。

### 剩余大鱼（排优先级）

1. **qsa:idx bf16/fp8 WMMA 打分**（数学改动）：t64 仅 8.7 TF，换 tensor core 打分 ~2×，
   128k 省 ~400 ms/16K ≈ 3.5%（32k 仅 ~0.8%）。需 ≥32K KLD 把关——bf16 基准
   `data/kld/bf16_c512.kld` 不在本机，且本机无 bf16 模型无法自生成，**须从 Linux 拷**。
   可顺带把 score 矩阵改 bf16 存储（省一半 select 读写流量），同属数学改动一起验。
2. **gdn:scan 深流水重构**（bit-exact，上限 ~3.5%）：P0 预取进寄存器藏 P1/P2 死时；
   LDS 57/64KB 无余量须寄存器倒手，数天工作量、收益不确定，排在 bf16 打分之后。
   正确性 harness 可直接复用 `tools/gdn_fused_proto.cu`（已验证 fused 与串行 wgrad
   逐 bit 一致）。

---

## 验证（每个改动都要过）

Windows 上可以直接跑 KLD：`gdec-win.exe` 支持 `--kld-base REF.kld --kld-chunks 64`
（`52_main.inc`），BF16 基准是 Linux 上的 `data/kld/bf16_c512.kld`，需要拷过去。

1. **KLD**：v2 基线 prefill KLD 为 0.10699（64×512，11b6847）。改动后不能超过 +0.002；
   方向 3、4 只改计算顺序，理论上应几乎不变。
2. **pp**：同一 maxctx、同一 chunk，32K 和 128K 各跑一次，取第二个 chunk 起的稳态值。
3. **w4b 回归**：方向 4 会改共享 kernel，w4b+overlay 的 KLD 必须和改前逐位相同，或在
   ±0.0005 内并说明原因。
4. 回到 Linux 后补跑 `tools/v2_verify.sh`（一键 PASS/FAIL），以及 `tools/bench_v2.sh`。

建议顺序：0 → 1（配置，当天就能有结论）→ 根据方向 0 的数据在 2 和 3+4 里挑。
如果方向 0 显示 Windows 的慢主要在某个具体 kernel，先追那个，它很可能比 2–5 加起来还大。

**验证口径补充（10-02 实机）**：

- 本机 32k@16384 只产出 1 个稳态 chunk，pp 结论一律以 128k（n=7 稳态 chunk）为准。
- bf16 基准文件到位前，用**自 A/B KLD** 作替代判定：改动前先存一份基线
  （`logs/kld_v2_before.kld`，8.1 GB，可复用），改后重跑对比，要求逐 bit 相同
  （mean_kld ≈ −0.000000、same_top=100.000）。方向 3、4 均以此法确认 bit-exact。
- microbench（`tools/moe_lut_test.cu`）必须在 GPU 空闲时跑，后台有 KLD/pp 在跑时数字会
  整体偏大（曾被污染出 up 42.6/down 23.3 ms 的假数据，作废）。
- 对比 KPROF 或 pp 时要剔除前 2 个 chunk（PLE 冷读），否则第一段会虚高几百 ms。
