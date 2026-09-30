#!/usr/bin/env python3
"""Encode a text file to token IDs (one per line) via build/tok_cli."""
import json
import subprocess
import sys

src, out, nmax, tokdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "models/tokenizer"
text = open(src, encoding="utf-8").read()
proc = subprocess.run(["build/tok_cli", tokdir], input=json.dumps({"op": "encode", "text": text}) + "\n",
                      capture_output=True, text=True, check=True)
ids = json.loads(proc.stdout.strip().splitlines()[-1])["ids"][:nmax]
with open(out, "w") as f:
    f.write("\n".join(map(str, ids)) + "\n")
print(f"wrote {len(ids)} ids -> {out}")
