#!/usr/bin/env python3
"""Judgement 4 helpers: prefix sharing + partial-page COW + post-restore hit.

Modes (engine wire protocol, port 8730):
  (default)  P1: 50000-token prompt (page 195 partial: 50000 = 195*256 + 80),
             max_tokens 16 -> end-of-turn checkpoint at 50016 tokens.
             P2: same 50000-token prefix + 512 fresh tokens, max_tokens 16
             -> prefix restore; continuing past token 50000 must COW the
             partial page shared with the checkpoint.
             Prints both D lines and P2's generated token ids.
  --fresh    run P2 only (against a snapshots-disabled service) and print
             its generated token ids -> compare with the COW-path run.
  --replay   run P1 again and print its D line (post-restart SSD restore
             check: n_cached = parts[10] should cover ~the whole prompt).
             NOTE: a checkpoint is only usable when strictly shorter than
             the prompt, so replaying P1 itself never hits its own
             checkpoint; use --chain for the post-restart SSD proof.
  --chain    T1: fresh 60000-token prompt, max_tokens 16 -> checkpoint
             k_60015-ish whose tail holds the generated tokens.
             T2: T1 prompt + T1's actual generated ids + 256 fresh tokens
             -> matches the FULL checkpoint (chain pages + tail) even when
             the RAM tier is empty, i.e. a true SSD restore after restart.
             Run once before the restart (RAM-tier baseline) and once after
             (SSD-tier proof); n_cached should be ~60015 both times.
"""
import socket
import sys
import time

HOST, PORT = "127.0.0.1", 8730
PREFIX = 50000
SUFFIX = 512
MAX_TOKENS = 16
EOS = [248046, 248044]

P1_IDS = [60000] + [60001 + (i % 4000) for i in range(PREFIX - 1)]
P2_IDS = P1_IDS + [64000 + (i % 2000) for i in range(SUFFIX)]


class Conn:
    def __init__(self):
        self.sock = socket.create_connection((HOST, PORT), timeout=30)
        self.buf = b""

    def line(self, timeout):
        self.sock.settimeout(timeout)
        while b"\n" not in self.buf:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise ConnectionError("engine closed the connection")
            self.buf += chunk
        out, self.buf = self.buf.split(b"\n", 1)
        return out.decode("utf-8", "replace")

    def gen(self, req, ids, max_tokens):
        t0 = time.time()
        head = f"GEN {req} {max_tokens} {len(EOS)}"
        body = " ".join(str(t) for t in EOS)
        payload = " ".join(str(i) for i in ids)
        self.sock.sendall(f"{head} {body} {len(ids)} {payload}\n".encode())
        toks, dline = [], None
        while True:
            ln = self.line(7200)
            if ln.startswith(f"T {req} "):
                toks.append(int(ln.split()[2]))
            elif ln.startswith(f"D {req} "):
                dline = ln
                break
        return dline, toks, time.time() - t0

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def show(name, req, ids):
    c = Conn()
    try:
        dline, toks, wall = c.gen(req, ids, MAX_TOKENS)
        parts = dline.split()
        cached = parts[10] if len(parts) > 10 and parts[2] not in ("error", "cancel") else "-"
        print(f"[{name}] {dline}")
        print(f"[{name}] n_cached={cached} wall={wall:.1f}s gen={toks}", flush=True)
        return dline, toks
    finally:
        c.close()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--fresh":
        show("P2-fresh", 202, P2_IDS)
        return 0
    if mode == "--replay":
        show("P1-replay", 203, P1_IDS)
        return 0
    if mode == "--chain":
        t1 = [70000] + [70001 + (i % 4000) for i in range(60000 - 1)]
        d1, g1 = show("T1", 210, t1)
        time.sleep(2)  # end-of-turn SSD capture commits asynchronously
        t2 = t1 + g1 + [76000 + (i % 2000) for i in range(256)]
        with open("/tmp/chain_t2.txt", "w") as f:
            f.write(" ".join(str(i) for i in t2))
        d2, _ = show("T2", 211, t2)
        cached2 = int(d2.split()[10]) if d2.split()[2] not in ("error", "cancel") else 0
        ok = (d1.split()[2] not in ("error", "cancel")
              and d2.split()[2] not in ("error", "cancel") and cached2 >= 60000)
        print(f"chain: T2 cached={cached2} (expect ~60015) -> "
              + ("PASS" if ok else "FAIL"), flush=True)
        return 0 if ok else 1
    if mode == "--chain-replay":  # post-restart: T2 only -> RAM tier is empty
        with open("/tmp/chain_t2.txt") as f:
            t2 = [int(x) for x in f.read().split()]
        d2, _ = show("T2-replay", 212, t2)
        cached2 = int(d2.split()[10]) if d2.split()[2] not in ("error", "cancel") else 0
        ok = d2.split()[2] not in ("error", "cancel") and cached2 >= 60000
        print(f"chain-replay (SSD tier): cached={cached2} (expect ~60015) -> "
              + ("PASS" if ok else "FAIL"), flush=True)
        return 0 if ok else 1
    d1, _ = show("P1", 200, P1_IDS)
    time.sleep(2)  # end-of-turn SSD capture commits asynchronously
    d2, t2 = show("P2-cow", 201, P2_IDS)
    ok1 = d1.split()[2] not in ("error", "cancel")
    ok2 = d2.split()[2] not in ("error", "cancel")
    cached2 = int(d2.split()[10]) if ok2 else 0
    print(f"cow-prefix: P1 ok={ok1} P2 ok={ok2} P2 cached={cached2} "
          f"(expect ~{PREFIX})", flush=True)
    if ok1 and ok2 and cached2 >= PREFIX - 512:
        print("PASS (COW path; verify gen ids match the --fresh reference)")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
