#!/usr/bin/env bash
# YaRN regression triple: baseline(main) save, factor1 compare, factor2 compare.
set -uo pipefail
cd /home/mark/Workspace/gfx1151-engine-yarn
BASEBIN=/home/mark/Workspace/engine-baseline/build/gdec
IDS=/tmp/yarn_short_ids.txt
REF=/tmp/yarn_main.kld

BIN="$BASEBIN" bash tools/kld_engine.sh --tokens "$IDS" "$REF" main_base
bash tools/kld_engine.sh "$REF" yarn_f1 GDEC_ROPE_FACTOR=1
bash tools/kld_engine.sh "$REF" yarn_f2 GDEC_ROPE_FACTOR=2 GDEC_ROPE_ORIGINAL_CTX=262144 GDEC_ROPE_BETA_FAST=32 GDEC_ROPE_BETA_SLOW=1
echo KLD_TRIPLE_DONE
