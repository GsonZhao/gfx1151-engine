#!/usr/bin/env bash
# pp_psweep_win.sh — M1 小 P 扫描（HANDOFF-PREFILL-ALL.md §2 M1）
# 固定 maxctx 139264，对 P ∈ {256 512 1024 2048 4096 16384}：prompt = 3×P（取
# data/qsa-oracle/131072.tokens 前缀），PREFILL_CHUNK=P，剔除预热 dummy 与第 1 个
# chunk（PLE 冷读），取第 2、3 个 chunk 均值。每档开 GDEC_KPROF=1 + GDEC_PROF=1。
# 引擎参数是启动期的，只能一档一进程。v2 模型（MODEL_FILE/NGRAM_FILE 可覆盖）。
# 用法: bash tools/pp_psweep_win.sh [P ...]   （默认六档全跑）
set -u
cd "$(dirname "$0")/.."

TOKENS_SRC=data/qsa-oracle/131072.tokens
MAXCTX=139264
PS="$@"
[ -z "$PS" ] && PS="256 512 1024 2048 4096 16384"

MODEL_ARGS=("${MODEL_FILE:-models/qwen38-flash-next-w4b.hgn}")
if [[ -z "${MODEL_FILE:-}" ]]; then
    MODEL_ARGS+=("models/qwen38-flash-next-w4b.overlay.hgn")
elif [[ -n "${NGRAM_FILE:-}" ]]; then
    MODEL_ARGS+=("$NGRAM_FILE")
fi

mkdir -p /tmp/psweep logs

for P in $PS; do
  NTOK=$((3 * P))
  TOK=/tmp/psweep_${NTOK}.tokens
  [ -f "$TOK" ] || head -n "$NTOK" "$TOKENS_SRC" > "$TOK"
  LOG="logs/m1_${TAG:-v2}_p${P}.log"
  env GDEC_KPROF=1 GDEC_PROF=1 \
      GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1 \
      GDEC_MOE_LT=1 GDEC_MOE_LT_BF16=1 GDEC_GR_BF16=1 \
      GDEC_GDN_STREAM=1 GDEC_GDN_WAVE=1 \
      GDEC_PREFILL_CHUNK=$P GDEC_GEMM_WMMA=1 GDEC_GDN_FUSED=1 \
      GDEC_INDEX_FUSED2=1 GDEC_PP_MOE_OUT=1 GDEC_INDEX_STREAM_SELECT=1 \
      GDEC_KVSNAP=1 GDEC_KVSNAP_MAX_GB=20 \
      build/gdec-win "${MODEL_ARGS[@]}" \
      --tokens-file "$TOK" --gen 1 --maxctx $MAXCTX >"$LOG" 2>&1
  rc=$?
  echo "== P=$P rc=$rc log=$LOG =="
done

# ---- 汇总（剔除预热行与 chunk 1，取 chunk 2、3 均值）----
python - $PS <<'EOF'
import re, sys, os
TAG = os.environ.get("TAG", "v2")
segs = ["gdn","qsa","moe:up","moe:down","moe:reduce","moe:shared","hc","ht_deq","gdn:scan","qsa:idx","qsa:flash"]

def parse(p):
    log = open(f"logs/m1_{TAG}_p{p}.log", encoding="utf-8", errors="replace").read()
    pre = re.findall(r"prefill: \d+ tokens in ([0-9.]+) s = ([0-9.]+) tok/s", log)
    kp  = re.findall(r"kprof base=(\d+) P=(\d+) segsum=([0-9.]+) ms[^\n]*", log)
    # 带分段的完整行
    kpl = [l for l in log.splitlines() if l.startswith("kprof base=")]
    prof = re.findall(r"prof: gemm_host=([0-9.]+)ms topk_wait=([0-9.]+)ms bucket\+h2d=([0-9.]+)ms\s+ple_host=([0-9.]+)ms ple_wait=([0-9.]+)ms", log)
    # 预热：prefill 行数 = nchunk+1 时第一行是 dummy
    nchunk = 3
    i = 1 if len(pre) == nchunk + 1 else 0
    wall = [float(t) for t,_ in pre[i+1:i+3]]          # chunk 2,3
    toks = [float(v) for _,v in pre[i+1:i+3]]
    ksel = kpl[i+1:i+3] if len(kpl) >= i+3 else kpl[-2:]
    psel = prof[i+1:i+3] if len(prof) >= i+3 else prof[-2:]
    seg_ms = {}
    segsum = 0.0
    for l in ksel:
        m = re.search(r"segsum=([0-9.]+) ms", l); segsum += float(m.group(1))
        for s in segs:
            m = re.search(r"(?:^|\| )\[?%s ([0-9.]+)" % re.escape(s), l)
            if m: seg_ms[s] = seg_ms.get(s, 0.0) + float(m.group(1))
    n = max(len(ksel), 1)
    segsum /= n
    for s in seg_ms: seg_ms[s] /= n
    wall_ms = sum(wall)/len(wall)*1000 if wall else 0.0
    tokps = sum(toks)/len(toks) if toks else 0.0
    ple = sum(float(x[4]) for x in psel)/max(len(psel),1) if psel else 0.0
    gh  = sum(float(x[0]) for x in psel)/max(len(psel),1) if psel else 0.0
    return tokps, wall_ms, segsum, seg_ms, ple, gh

hdr = ["P","tok/s","wall ms","segsum","gap%"] + segs + ["ple_wait","gemm_host"]
print(("\t".join(hdr)))
for p in sys.argv[1:]:
    tokps, wall_ms, segsum, seg_ms, ple, gh = parse(int(p))
    gap = (wall_ms - segsum) / wall_ms * 100 if wall_ms else 0.0
    row = [p, f"{tokps:.1f}", f"{wall_ms:.1f}", f"{segsum:.1f}", f"{gap:.1f}"]
    row += [f"{seg_ms.get(s,0.0):.1f}" for s in segs]
    row += [f"{ple:.1f}", f"{gh:.1f}"]
    print("\t".join(row))
EOF
