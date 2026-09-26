#!/usr/bin/env python3
"""ms_download.py — ModelScope 仓库多线程下载器（断点续传 + 大小校验）

用法:
  python ms_download.py <model_id> <dest_dir> [jobs]

默认 8 线程并发下载单文件,同时最多 4 个文件并行;已存在且大小一致的
文件跳过,分片文件支持 HTTP Range 续传。
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://modelscope.cn/api/v1/models/{mid}/repo/files?PageSize=500"
URL = "https://modelscope.cn/api/v1/models/{mid}/repo?Revision=master&FilePath={path}"


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ms-dl/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def list_files(mid):
    d = fetch_json(API.format(mid=mid))
    files = [(f["Path"], int(f.get("Size", 0))) for f in d["Data"]["Files"]]
    return files


def download_one(mid, path, size, dest):
    out = os.path.join(dest, path.replace("/", os.sep))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = URL.format(mid=mid, path=urllib.request.quote(path))
    for attempt in range(6):
        try:
            done = os.path.getsize(out) if os.path.exists(out) else 0
            if done == size:
                return path, "skip"
            if done > size:
                os.remove(out)
                done = 0
            req = urllib.request.Request(url, headers={
                "User-Agent": "ms-dl/1.0",
                **({"Range": f"bytes={done}-"} if done else {}),
            })
            with urllib.request.urlopen(req, timeout=120) as r, \
                 open(out, "ab" if done else "wb") as f:
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(out) == size:
                return path, "ok"
            raise IOError(f"size mismatch {os.path.getsize(out)} != {size}")
        except Exception as e:  # noqa: BLE001
            wait = min(30, 2 ** attempt)
            print(f"[retry {attempt+1}] {path}: {e}; {wait}s", flush=True)
            time.sleep(wait)
    return path, "FAIL"


def main():
    mid, dest = sys.argv[1], sys.argv[2]
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    files = list_files(mid)
    total = sum(s for _, s in files)
    print(f"{mid}: {len(files)} files, {total/1e9:.1f} GB -> {dest}", flush=True)
    os.makedirs(dest, exist_ok=True)
    t0 = time.time()
    fails = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(download_one, mid, p, s, dest): p for p, s in files
                if s > 0}
        done_bytes = sum(os.path.getsize(os.path.join(dest, p.replace('/', os.sep)))
                         for _, p in [(x, x) for x in []])  # noqa
        done_bytes = 0
        for fut in as_completed(futs):
            path, status = fut.result()
            if status == "FAIL":
                fails.append(path)
            sz = dict(files).get(path, 0)
            done_bytes += sz
            dt = time.time() - t0
            print(f"[{status}] {path} ({sz/1e9:.2f} GB) "
                  f"| {done_bytes/1e9:.1f}/{total/1e9:.1f} GB "
                  f"| {dt/60:.1f} min | {done_bytes/max(dt,1)/1e6:.1f} MB/s",
                  flush=True)
    if fails:
        print(f"FAILED: {fails}", file=sys.stderr)
        sys.exit(1)
    print("ALL_FILES_OK", flush=True)


if __name__ == "__main__":
    main()
