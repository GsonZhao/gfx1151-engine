#!/usr/bin/env python3
"""Judgement 2: non-prefix reuse of two NON-EMPTY slots (PARALLEL=2, 2048-page pool).

Phase 1 (fill): two 102400-token (400-page) GENs with distinct id patterns,
run sequentially on separate connections -> slot0 and slot1 both hold ~400
pages of live state.

Phase 2 (concurrent reuse):
  R1: 23024-token prompt, max_tokens 16 -> admission target 90 pages,
      non-prefix -> resets the LRU 400-page slot.
  R2: 460784-token prompt, max_tokens 16 -> admission target 1800 pages,
      resets the other slot.
  90 + 1800 = 1890 < 2048 -> both must be admitted. The pre-fix accounting
  double-counted the reset slot's freed 400 pages and could reject R2.

Pass criteria: both GENs succeed; engine log shows each req landing on a
slot with ~1024xx live tokens plus a "no live prefix" line (proving reuse
of non-empty slots, not empty-slot / live-prefix hits); service answers a
small request afterwards.
"""
import socket
import sys
import threading
import time

HOST, PORT = "127.0.0.1", 8730
PAGE = 256
FILL_TOKENS = 400 * PAGE        # 102400
R1_TOKENS = 90 * PAGE - 16      # 23024  (+16 decode = 90 pages exactly)
R2_TOKENS = 1800 * PAGE - 16    # 460784 (+16 decode = 1800 pages exactly)
MAX_TOKENS = 16
EOS = [248046, 248044]


def make_ids(base, n):
    # distinct first token per prompt -> guaranteed non-prefix vs the others
    return [base] + [base + 1 + (i % 4000) for i in range(n - 1)]


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
        dline = None
        while True:
            ln = self.line(7200)
            if ln.startswith(f"D {req} "):
                dline = ln
                break
        return dline, time.time() - t0

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def run(name, req, ids, max_tokens, results):
    c = Conn()
    try:
        dline, wall = c.gen(req, ids, max_tokens)
        results[name] = (dline, wall)
        print(f"  {name}: {dline}  wall={wall:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        results[name] = (f"EXC {e}", -1)
        print(f"  {name}: EXCEPTION {e}", flush=True)
    finally:
        c.close()


def ok(dline):
    parts = dline.split()
    return len(parts) >= 3 and parts[0] == "D" and parts[2] not in ("error", "cancel")


def main():
    results = {}
    print(f"== phase 1: fill both slots ({FILL_TOKENS} tokens each) ==", flush=True)
    run("fill-A", 101, make_ids(10000, FILL_TOKENS), MAX_TOKENS, results)
    run("fill-B", 102, make_ids(20000, FILL_TOKENS), MAX_TOKENS, results)
    for k in ("fill-A", "fill-B"):
        if not ok(results[k][0]):
            print(f"FAIL: {k} did not complete: {results[k][0]}")
            return 1

    time.sleep(2)  # let slot_leave settle (LRU order deterministic)
    print(f"== phase 2: concurrent non-prefix reuse "
          f"(90-page reset + 1800-page reset, pool 2048) ==", flush=True)
    t0 = time.time()
    t1 = threading.Thread(target=run, args=("R1-90p", 103, make_ids(30000, R1_TOKENS), MAX_TOKENS, results))
    t2 = threading.Thread(target=run, args=("R2-1800p", 104, make_ids(40000, R2_TOKENS), MAX_TOKENS, results))
    t1.start()
    time.sleep(1.0)  # R1 takes the first admission ticket
    t2.start()
    t1.join()
    t2.join()
    print(f"  concurrent wall={time.time() - t0:.1f}s", flush=True)

    r1_ok, r2_ok = ok(results["R1-90p"][0]), ok(results["R2-1800p"][0])
    print(f"  R1 admitted+completed: {r1_ok}; R2 admitted+completed: {r2_ok}",
          flush=True)

    print("== phase 3: post-test health ==", flush=True)
    run("health", 105, make_ids(50000, 64), MAX_TOKENS, results)
    health_ok = ok(results["health"][0])

    if r1_ok and r2_ok and health_ok:
        print("PASS: both non-empty slots reused (90p+1800p <= 2048 admitted); "
              "check engine log for '-> slot N (1024.. live tokens)' + 'no live prefix'")
        return 0
    print("FAIL: see per-request D lines above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
