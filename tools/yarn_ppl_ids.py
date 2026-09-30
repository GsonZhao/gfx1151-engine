#!/usr/bin/env python3
"""Build token-ID files for long-context PPL runs.

long:  filler tiled to F tokens + target (T ids) -> positions >= F are scored tail
short: filler F0 ids + same target -> position-baseline for the same tail
"""
import json
import subprocess
import sys

F = int(sys.argv[1]) if len(sys.argv) > 1 else 350000   # filler tokens for the deep run
T = sys.argv[2] if len(sys.argv) > 2 else "/tmp/yarn_short_ids.txt"

FILLER = ("Field report {i:06d}: surveyors mapped another section of the valley, "
          "recording soil samples, creek levels, fence repairs, and weather notes. "
          "The findings were catalogued and archived without incident.\n")


def tok_ids(text):
    proc = subprocess.run(["build/tok_cli", "models/tokenizer"],
                          input=json.dumps({"op": "encode", "text": text}) + "\n",
                          capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])["ids"]


target = [int(x) for x in open(T).read().split()]
block = tok_ids(FILLER.format(i=0))
k = F // len(block)
filler = []
for i in range(k):
    filler += tok_ids(FILLER.format(i=i)) if False else block  # same block tiled
filler = filler[:F]
print(f"filler={len(filler)} target={len(target)}", file=sys.stderr)

def dump(path, ids):
    with open(path, "w") as f:
        f.write("\n".join(map(str, ids)) + "\n")

dump("/tmp/ppl_long_ids.txt", filler + target)
dump("/tmp/ppl_short_ids.txt", filler[: len(target)] + target)
print(f"long={len(filler)+len(target)} short={2*len(target)}", file=sys.stderr)
