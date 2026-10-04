#!/usr/bin/env bash
# Windows manual driver for the toolcall/agent tcache scenarios (tcache_verify.sh
# is Linux-only: ss/start_hgn.sh --check). Engine :8732, API cache-ON :8733,
# cache-OFF :8734; runs --only toolcall then --only agent.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
A1=''; A2=''; EPID=''
cleanup() {
  kill $EPID $A1 $A2 2>/dev/null
  for p in $EPID $A1 $A2; do [ -n "$p" ] && taskkill //PID $p //F //T >/dev/null 2>&1; done
}
trap cleanup EXIT

env GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1 GDEC_MOE_LT=1 \
    GDEC_MOE_LT_BF16=1 GDEC_GR_BF16=1 GDEC_GDN_STREAM=1 GDEC_GDN_WAVE=1 \
    GDEC_PREFILL_CHUNK=16384 GDEC_GEMM_WMMA=1 GDEC_GDN_FUSED=1 \
    GDEC_INDEX_FUSED2=1 GDEC_PP_MOE_OUT=1 GDEC_INDEX_STREAM_SELECT=1 \
    GDEC_KVSNAP=0 GDEC_RCKPT_MIN=200 \
    build/gdec-win models/qwen38-flash-next-w4b.hgn \
    models/qwen38-flash-next-w4b.overlay.hgn \
    --serve --host 127.0.0.1 --port 8732 --maxctx 32768 \
    > logs/agent_engine.log 2>&1 &
EPID=$!
ok=0
for i in $(seq 1 150); do
  grep -q 'serve: listening' logs/agent_engine.log 2>/dev/null && { ok=1; break; }
  kill -0 $EPID 2>/dev/null || { tail -n 15 logs/agent_engine.log; exit 1; }
  sleep 2
done
[ "$ok" = 1 ] || { echo "引擎启动超时"; exit 1; }

build/gdec-api-win.exe --tokenizer models/tokenizer --engine 127.0.0.1:8732 \
  --host 127.0.0.1 --port 8733 --context 32768 > logs/agent_api_on.log 2>&1 &
A1=$!
GDEC_API_TOKCACHE=0 build/gdec-api-win.exe --tokenizer models/tokenizer \
  --engine 127.0.0.1:8732 --host 127.0.0.1 --port 8734 --context 32768 \
  > logs/agent_api_off.log 2>&1 &
A2=$!
for port in 8733 8734; do
  for i in $(seq 1 30); do
    netstat -an | grep -E "[:.]${port}\s+.*LISTENING" >/dev/null 2>&1 && break
    sleep 1
  done
done
grep 'ckpt tokens' logs/agent_api_on.log || echo "WARN: API 日志无 ckpt tokens 行"

rc=0
echo "==== toolcall 场景 ===="
python tools/tcache_verify.py --on 8733 --off 8734 --max-tokens 3000 --only toolcall || rc=1
echo "==== agent 场景 ===="
python tools/tcache_verify.py --on 8733 --off 8734 --max-tokens 3000 --only agent || rc=1
echo "==== 引擎 ckpt/rckpt 摘录 ===="
grep -E "ckpt: saved|rckpt: restored|rckpt evicted" logs/agent_engine.log | tail -n 12
echo "AGENT VERIFY: $([ $rc = 0 ] && echo PASS || echo FAIL)"
exit $rc
