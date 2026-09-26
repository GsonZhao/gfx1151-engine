"""recover_hgn.py — resume a flashnext2hgn conversion that died during the
final PLE-table merge (OSError 28 / disk full).

State assumed:
  models/<name>.hgn          header + all base blobs written; PLE blob
                             partially appended by shutil.copyfileobj
  models/<name>.hgn.ple-tmp  complete quantized PLE blob (intact)

Steps:
  1. Parse the hgn header, locate the DT_FP8 PLE record -> its offset is the
     exact pre-PLE file size (written by begin() itself).
  2. Truncate the base file back to that offset, then stream-copy the full
     ple-tmp blob (byte-exact resume), pad to 64B, verify size == header
     `total`, delete ple-tmp.
  3. Redo the post-base stages of convert(): MTP sidecar, vision tower,
     tokenizer copy, start.sh.
"""
import os, sys, struct, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flashnext2hgn as m
import numpy as np

MODEL_DIR = r"C:\models-src\Qwen3.8-Flash-Next"
OUTDIR = os.path.join(os.path.dirname(HERE), "models")
NAME = "qwen38-flash-next-w4b"
BASE = os.path.join(OUTDIR, NAME + ".hgn")
TMP = BASE + ".ple-tmp"
HDR, REC = m.HDR, m.REC


def log(*a):
    print(*a, flush=True)


# ---- 1. parse header, find PLE record -------------------------------------
with open(BASE, "rb") as f:
    hdr = f.read(HDR)
    magic, ver, n, _, _, data_off, total = struct.unpack_from("<4sIIIQQQ", hdr, 0)
    assert magic == b"HGN1", magic
    ple_rec = None
    for i in range(n):
        r = f.read(REC)
        name = r[:96].split(b"\0", 1)[0].decode()
        dt, nd = struct.unpack_from("<II", r, 96)
        off, size, _ = struct.unpack_from("<QQQ", r, 136)
        if dt == m.DT_FP8:
            assert ple_rec is None, "multiple FP8 records?"
            ple_rec = (name, off, size)
assert ple_rec, "no FP8 (PLE) record found"
ple_name, S_pre, ple_size = ple_rec
cur = os.path.getsize(BASE)
tmp_size = os.path.getsize(TMP)
log(f"PLE record: {ple_name} off={S_pre} size={ple_size} ({ple_size/2**30:.1f} GiB)")
log(f"base now {cur} (partial copy {cur - S_pre}), tmp {tmp_size}")
assert tmp_size == ple_size, f"tmp size {tmp_size} != planned {ple_size}"
assert S_pre <= cur <= S_pre + ple_size

# ---- 2. truncate + resume copy ---------------------------------------------
t0 = time.time()
with open(TMP, "rb") as src, open(BASE, "r+b") as dst:
    dst.truncate(S_pre)
    dst.seek(S_pre)
    done = 0  # full recopy from tmp[0]; truncate wiped any partial tail
    while True:
        buf = src.read(1 << 28)
        if not buf:
            break
        dst.write(buf)
        done += len(buf)
        log(f"  ple copy {done/2**30:.1f}/{ple_size/2**30:.1f} GiB "
            f"({time.time()-t0:.0f}s)")
    pad = (-ple_size) % 64
    if pad:
        dst.write(b"\0" * pad)
    final = dst.tell()
assert final == total, f"final size {final} != header total {total}"
os.unlink(TMP)
log(f"base file COMPLETE: {final} bytes == header total, ple-tmp removed")

# ---- 3. post-base stages (replica of convert()) ----------------------------
cfg = m.check_config(__import__("json").load(
    open(os.path.join(MODEL_DIR, "config.json")))) or None
cfg = __import__("json").load(open(os.path.join(MODEL_DIR, "config.json")))
m.check_config(cfg)
log(f"scanning {MODEL_DIR} ...")
table = m.scan_model_dir(MODEL_DIR)
log(f"  {len(table)} source tensors")

ple_shards, base_src, vis_src, mtp_src = {}, {}, {}, {}
for src_name, (st, dt, shape) in table.items():
    if src_name.startswith(m.VIS_PREFIX):
        vis_src["visual." + src_name[len(m.VIS_PREFIX):]] = (st, dt, shape, src_name)
        continue
    mm = m.map_lm_name(src_name)
    assert mm is not None, src_name
    if mm.startswith("layers.1.ple.ple_embedding.ngram_embedding.shard_"):
        continue
    if mm.startswith("mtp."):
        mtp_src[mm] = (st, dt, shape, src_name)
    else:
        base_src[mm] = (st, dt, shape, src_name)

# ---- MTP sidecar ----
side_path = os.path.join(OUTDIR, f"{NAME}-mtp.hgn")
ws = m.HgnWriter(side_path, NAME + "-mtp")
for mm in sorted(m.MTP_Q8):
    assert mm in mtp_src, f"missing MTP tensor: {mm}"
    _, _, shape, _ = mtp_src[mm]
    cols = shape[-1]
    rows = int(np.prod(shape)) // cols
    ws.plan(mm, m.DT_Q8G64, shape, m.q8g64_size(rows, cols))
ws.begin()
for mm in sorted(m.MTP_Q8):
    st, _, shape, src = mtp_src[mm]
    ta = time.time()
    ws.write(m.quant_q8g64(st.f32(src).reshape(-1, shape[-1])))
    log(f"  mtp8 {mm} {shape} in {time.time()-ta:.1f}s")
ws.finish()
log(f"wrote {side_path} ({os.path.getsize(side_path)/2**30:.2f} GiB)")

# ---- vision ----
vis_path = os.path.join(OUTDIR, f"{NAME}-vision.hgn")
wv = m.HgnWriter(vis_path, NAME + "-vision")
for mm in sorted(vis_src.keys()):
    _, sdt, shape, _ = vis_src[mm]
    if mm == "visual.patch_embed.proj.weight":
        shape = (shape[0], int(np.prod(shape[1:])))
    wv.plan(mm, m.DT_BF16, shape, int(np.prod(shape)) * 2)
wv.begin()
for mm in sorted(vis_src.keys()):
    st, sdt, shape, src = vis_src[mm]
    assert sdt == "BF16", f"{mm}: {sdt}"
    _, _, buf = st.raw(src)
    wv.write(bytes(buf))
wv.finish()
log(f"wrote {vis_path} ({os.path.getsize(vis_path)/2**30:.2f} GiB)")

# ---- tokenizer ----
tok_dir = os.path.join(OUTDIR, "tokenizer")
os.makedirs(tok_dir, exist_ok=True)
for f in ("tokenizer.json", "vocab.json", "merges.txt",
          "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "preprocessor_config.json"):
    src = os.path.join(MODEL_DIR, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(tok_dir, f))
log(f"tokenizer files -> {tok_dir}")

# ---- start script ----
engine_dir = os.path.join(os.path.dirname(HERE), "build")
try:
    sp = m.write_start_script(OUTDIR, NAME, True, True, engine_dir)
    log(f"start script -> {sp}")
except Exception as e:
    log(f"start.sh skipped ({e}) — start_win.exe is unaffected")

log("RECOVERY_DONE")
