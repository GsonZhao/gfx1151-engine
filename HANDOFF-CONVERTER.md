# 交接：集成转换器（原始权重 → 高质量 hgn，有/无 imatrix）的最终验证

写于 2026-09-26 11:50。接手的助手请从头读到尾再动手。回复用户用中文。

## 1. 目标

用户原话："把转换脚本做一个集成吧，用户可以轻松的从原始权重转换出 hgn，有无 imatrix 都能转。"

- 代码已经写完，本地合成自测也已 PASS。
- 剩下的是在远程机器上用真实权重跑两遍完整验证（有 imatrix、无 imatrix），把结果填进文档，再本地提交。
- 这是夜间无人值守任务：**不要问用户问题，也不要自己扩大范围**。

## 2. 背景（只需知道这些）

- 高质量 hgn 在凌晨已经用两个"在旧文件上重建"的工具做出来了：`tools/hgn_hq.py`（8-bit dense overlay）和 `tools/hgn_q4i.py`（imatrix 专家基座）。
  - KLD 0.0558，top1 92.41%，PPL 3.329，MTP commit/round 3.76。
  - 文件在远程 `~/Models/hq/`。**这些文件不要动。**
- 集成转换器 `tools/flashnext2hgn.py` 把上面的流程合成一步，直接从 safetensors 生成全部文件：
  - `<name>.hgn`、`<name>.overlay.hgn`、`<name>-mtp.hgn`、`-vision.hgn`、`tokenizer/`、`start.sh`。
  - `--imatrix` 是可选参数。`--classic` 保留旧行为，与旧转换器逐字节相同。
- 之前两次真实验证都失败了，原因在路由专家的码表：

  | 码表 | KLD |
  |---|---|
  | 旧转换器码表（接近均匀） | 0.0615，FAIL |
  | 每张量训练（现在的 `--expert-codebook trained`） | 0.0587（top1 91.80%、PPL 3.355、spec 3.78） |
  | 现在的默认 `UNIVERSAL_EXPERT_CB`（生产文件的固定码表形状） | **预期 ≈ 0.0558**，这就是要验证的 |

- 第三次验证（universal 码表）11:26 启动后被用户叫停：用户不想白天等。输出已删除，没有结果。

## 3. 本地未提交的文件（`git status`）

要提交的：

| 文件 | 状态 | 内容 |
|---|---|---|
| `tools/flashnext2hgn.py` | M | HQ 默认、`--imatrix`、`--expert-codebook universal\|trained`、overlay、抽查 |
| `tools/hgn_hq.py` | M | 改为从 flashnext2hgn 导入 q8g32 / OVL_BF16 |
| `tools/pp_prod.sh` | M | 小改（传参） |
| `tools/convert_selftest.py` | ?? | 合成自测，本地已 PASS |
| `tools/convert_verify.sh` | ?? | 远程一键验证，打印 PASS/FAIL |
| `HGN-HQ.md` | ?? | 文档；第 4 节表里 `@CONV_IM@` / `@CONV_NO@` 两行待填 |
| `GGUF.md`、`PORTING-WINDOWS.md` | M | 各加了一段指向 HGN-HQ.md |

**绝对不要 stage**：

- 只有 CRLF 差异的文件：`.zcodeignore CONVERT.md CONVERT_EN.md HGN-FORMAT.md HGN-FORMAT_EN.md NGRAM.md NGRAM_EN.md README.md README_EN.md src/api/reqstat.cpp`。
- 未跟踪的笔记：`A1-VERIFY.md HANDOFF-PHASE-A.md KVSNAP-REVIEW.md MEMORY-MODEL-CONCURRENCY.md SSD-KVSNAP-PHASE2.md STRIX-HALO-PORTING.md`，以及本文件 `HANDOFF-CONVERTER.md`。

## 4. 远程机器

- 主机 192.168.1.12，用户 mark，目录 `~/Workspace/gfx-1151-kvsnap`。这个目录不是 git 仓库，以本地代码为准，同步方式见下。
- HOME 是 `/home/mark`。numpy 在 `~/Workspace/pylib`，convert_verify.sh 会自动设置 PYTHONPATH。
- SSH 密码：开始前向用户要一次，写进 VM 的 `/tmp/ap.sh`（`#!/bin/sh` + `echo '<密码>'`，chmod 700）。**不要打印，不要写进任何文件、文档或记忆。** 连接方式：
  ```bash
  export SSH_ASKPASS=/tmp/ap.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
  S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 mark@192.168.1.12"
  timeout 60 $S 'bash -s' < /tmp/script.sh      # 远程命令一律写成脚本文件再这样执行
  ```
- 同步（本地 → 远程），**只能在远程没有任务运行时做**：
  ```bash
  cd <本地仓库> && tar cf /tmp/x.tar tools/flashnext2hgn.py tools/convert_selftest.py tools/convert_verify.sh tools/hgn_hq.py tools/pp_prod.sh
  timeout 60 $S 'cd ~/Workspace/gfx-1151-kvsnap && tar xf -' < /tmp/x.tar
  ```
  11:20 已同步过一次。本地 `tools/flashnext2hgn.py` 的 md5 是 `b55d6134d1a2d2e0219c9a892dbaf196`，远程相同就不用再同步。
- 已经存在、可以直接用的远程文件：
  - `build/gdec-hq2`：带 8-bit overlay gemv 的引擎。**用它，不要用 `build/gdec`**，后者是 09-25 的旧版。
  - `logs/flashnext2hgn_ref.py`：旧转换器，供 `--classic` 对照。
  - `logs/cvrun.sh`：有 imatrix 那一轮的启动器。
  - `logs/cv_trained/`：trained 那一轮的日志存档。

## 5. 安全规则（违反过就出过事）

1. **远程 `models/` 里的东西只能读**，不能改、删、改名。新输出只放 `~/Models/hq/` 下。
2. **避免内核死锁**。09-25 在 page cache 塞满时加载过模型，导致 amdgpu SVM 自锁，出现杀不掉的 D 状态进程；没有 sudo，只能等用户重启。
   - 每次轮询都检查 `journalctl -k -b | grep -c svm_range_cpu_invalidate_pagetables`。现在是 0；**一旦变大，立刻停手**，什么都别再跑，早上报告用户。
   - 永远不要用 `GDEC_PREFILL_CHUNK=32768` 跑 GGUF。
   - 不要在 GPU 测试的同时跑 CPU 重的转换。convert_verify.sh 内部是串行的，也会在上 GPU 前做 fadvise 清 cache，**两轮之间也必须串行**。
3. 后台任务用 `(setsid nohup bash X > /dev/null 2>&1 < /dev/null &)` 启动，并且写在脚本文件里。每次工具调用控制在约 175 s 以内（`sleep` ≤ 170）。
4. **引擎守卫**：convert_verify.sh 发现任何进程命令行含 `gdec` 就拒绝运行。所以启动脚本里写 `A=gd; BIN=build/${A}ec-hq2`，不要让 SSH 命令行里出现字面量 "gdec"。
5. **`pkill -f` 必须用方括号写法**，例如 `pkill -f '[c]onvert_verify.sh'`。不这样写，它会匹配到 SSH 执行的 bash 本身，把自己的连接杀掉。
6. 运行中的脚本不要同步覆盖。

## 6. 要做的事（按顺序）

### 6.1 第一轮：有 imatrix，约 2 小时

```bash
# /tmp/run_im.sh
cd ~/Workspace/gfx-1151-kvsnap || exit 1
echo "svm=$(journalctl -k -b | grep -c svm_range_cpu_invalidate_pagetables)"
pgrep -af '[g]dec|[f]lashnext2hgn|[c]onvert_verify|[l]lama-' && { echo BUSY; exit 1; }
D=$(realpath -m ~/Models/hq/conv); [[ $D == /home/mark/Models/hq/conv ]] || exit 1
rm -rf "$D"; df -h ~/Models/hq | tail -1          # 需要 >= 135 GiB（11:50 时剩 414G）
(setsid nohup bash logs/cvrun.sh > /dev/null 2>&1 < /dev/null &)
sleep 3; pgrep -af '[c]onvert_verify'
```

`logs/cvrun.sh` 的内容（已在远程）：

```bash
cd ~/Workspace/gfx-1151-kvsnap
A=gd
export BIN=build/${A}ec-hq2 REF_CONV=logs/flashnext2hgn_ref.py
bash tools/convert_verify.sh > logs/convert_verify.out 2>&1
echo "CV_RC=$?" >> logs/convert_verify.out
```

轮询脚本：

```bash
cd ~/Workspace/gfx-1151-kvsnap
tail -c 3000 logs/convert_full.log | tr '\r' '\n' | grep -E 'base \[|check|FAIL|wrote' | tail -2
grep -E 'CV_RC|^PASS$|^FAIL|KLD=|commit/round' logs/convert_verify.out | tail -4
echo "svm=$(journalctl -k -b | grep -c svm_range_cpu_invalidate_pagetables)"
```

各步骤的时间和预期：

| 步骤 | 用时 | 预期结果 |
|---|---|---|
| 1. 合成自测 | ~40 s | 全部 PASS |
| 2. `--only-overlay` | ~45 s | 与 `qwen38-flash-next-w4b.overlay-q8.hgn` 逐字节相同 |
| 3. 完整转换 | ~90 min | 每个 `check` 行的倍数 > 1 |
| 4. KLD | — | ≤ 0.060，预期约 0.056 |
| 5. MTP spec | — | commit/round ≥ 3.0，预期约 3.7 |
| 结尾 | — | `PASS` 和 `CV_RC=0` |

从日志里记下 KLD、top1、PPL、commit/round。

### 6.2 第二轮：无 imatrix，前一轮结束后再开始

- 先 `mv logs/convert_verify.out logs/convert_verify.imat.out`，保存第一轮日志。其它 `logs/convert_*.log`、`kld_conv_kld.log`、`conv_spec.log` 会被覆盖，也先拷一份到 `logs/cv_imat/`。
- 为了省盘，删掉 `~/Models/hq/conv`（仍然要做路径守卫：realpath 必须等于 `/home/mark/Models/hq/conv`）。
- 启动器和 6.1 相同，只是环境变量多两个：`IMATRIX=none OUT=$HOME/Models/hq/conv-noimat`。可以新写一个 `logs/cvrun_noimat.sh`。
- KLD 上限自动变成 0.066。结果没有先例，大概在 0.058–0.064。
- 结束后删掉 `~/Models/hq/conv-noimat`。两轮的日志都要保留。

### 6.3 失败时怎么办

- 看 `logs/convert_verify.out` 里的 FAIL 行，以及对应的 `logs/convert_full.log`、`logs/kld_conv_kld.log`、`logs/conv_spec.log`。
- **不要循环重跑**（用户明确反感）。每轮最多允许一次针对明确 bug 的修复重跑，否则停下来，早上报告。
- 如果有 imatrix 那一轮的 KLD 明显高于 0.058，先比对码表：
  - 输出基座里专家张量前 64 字节应该等于 `UNIVERSAL_EXPERT_CB`；
  - 生产 `models/qwen38-flash-next-w4b.hgn` 的专家码表归一化后应该几乎相同。
  - 只调查，不要改默认。

### 6.4 填文档，然后本地提交

- `HGN-HQ.md` 第 4 节：用两轮的 KLD / top1 / PPL / commit/round 替换 `@CONV_IM@`、`@CONV_NO@`。其余列（pp/decode）没测就留空，不要另外去跑基准。
- 在"码表"小节的表里，确认 universal 那行的数字与实测一致，不一致就改成实测值。
- 本地提交（VM 推不了，用户自己从 Windows push）：
  ```bash
  cd <本地仓库>
  git add tools/flashnext2hgn.py tools/hgn_hq.py tools/pp_prod.sh tools/convert_selftest.py tools/convert_verify.sh
  export GIT_AUTHOR_NAME=Mark GIT_AUTHOR_EMAIL=2432896620@qq.com GIT_COMMITTER_NAME=Mark GIT_COMMITTER_EMAIL=2432896620@qq.com
  git commit -m "tools: integrated converter flashnext2hgn.py -> HQ hgn (8-bit dense overlay, weighted q4cp experts, optional imatrix, universal expert codebook); convert_selftest/convert_verify"
  git add HGN-HQ.md GGUF.md PORTING-WINDOWS.md
  git commit -m "docs: HGN-HQ.md (high-quality hgn, one-step converter, results)"
  git status --short     # 确认 §3 列出的"不要 stage"的文件都没进提交
  ```
  提交前在本地再跑一遍自测，确认结尾是 `CONVERT_SELFTEST PASS`（约 1 分钟）：
  `git show e2f4d9f:tools/flashnext2hgn.py > /tmp/fn2h_ref.py && python3 tools/convert_selftest.py --ref /tmp/fn2h_ref.py`

## 7. 早上给用户的报告（中文，简短）

- 两轮各自的 PASS/FAIL，以及 KLD / top1 / PPL / commit/round，并与 hgn_q4i 版的 0.0558 对比。
- 本地新增了哪几个提交（提交号），提醒用户从 Windows push。
- 转换器用法，两行即可：
  ```bash
  PYTHONPATH=~/Workspace/pylib python3 tools/flashnext2hgn.py ~/Models/Qwen3.8-Flash-Next --out ~/Models/hq/xxx \
      [--imatrix ~/Models/BF16/Qwen3.8-Flash-Next-BF16/imatrix_unsloth.gguf_file]
  ```
  32 核约 1.5 小时，输出约 125 GiB，生成的 `start.sh` 可直接启动。
- 远程磁盘上删了什么、留了什么。svm 计数是否仍为 0。
