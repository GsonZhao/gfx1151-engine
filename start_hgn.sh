#!/usr/bin/env bash
# hgn 权重启动器（Linux）：models/ 下的 qwen38-flash-next-*.hgn。llama.cpp GGUF 权重请用 start_gguf.sh。
# 前台运行，Ctrl+C 同时停止本次启动的 API 和引擎；--check 只检查配置、不启动。
# 路径与部署参数见 service.conf（"hgn 权重"一段），同名环境变量优先。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LAUNCHER=start_hgn.sh
source "$ROOT/tools/serve_common.sh"
serve_init "$@"

for f in "$MODEL_FILE" "$NGRAM_FILE" "$OVERLAY_FILE" "$MTP_FILE" "$VISION_FILE"; do
  [[ -z "$f" || "$f" == *.hgn ]] || fail "$LAUNCHER 只接受 .hgn 权重：$f（GGUF 请用 bash start_gguf.sh）"
done
need 主模型 "$MODEL_FILE"
NGRAM_FILE=${NGRAM_FILE:-$MODEL_FILE}
need 'n-gram 表' "$NGRAM_FILE" 'w4b 与主模型同一文件；v2 指向 *-ngram.hgn'
# overlay 可选：文件不在只警告并跳过（例如换成 v2 权重后忘了改配置）。
if [[ -n "$OVERLAY_FILE" && ! -r "$OVERLAY_FILE" ]]; then
  echo "警告：找不到 overlay：$OVERLAY_FILE，已跳过（v2 权重不用 overlay；不想看到这条可设 OVERLAY_FILE=\"\"）" >&2
  OVERLAY_FILE=
fi
[[ -z "$MTP_FILE" ]] || need 'MTP 草稿' "$MTP_FILE" '设 MTP_FILE="" 退回 overlay 内置草稿头'
[[ -z "$VISION_FILE" ]] || need 视觉塔 "$VISION_FILE" '纯文本可设 VISION_FILE=""'
MISSING_HINT='hgn 权重由 tools/flashnext2hgn.py 转换生成（见 CONVERT.md）；手上是 llama.cpp GGUF 请用 bash start_gguf.sh（见 GGUF.md）。'

FORMAT=hgn
MAIN_MODEL=$MODEL_FILE
MODEL_ARGS=("$MODEL_FILE")
[[ -z "$OVERLAY_FILE" ]] || MODEL_ARGS+=("$OVERLAY_FILE")
# 独立 n-gram 文件（v2）排在 overlay 之后、MTP 之前；与主模型同一文件（w4b）则不重复传。
[[ "$NGRAM_FILE" -ef "$MODEL_FILE" ]] || MODEL_ARGS+=("$NGRAM_FILE")
[[ -z "$MTP_FILE" ]] || MODEL_ARGS+=("$MTP_FILE")
VISION=$VISION_FILE
serve_run
