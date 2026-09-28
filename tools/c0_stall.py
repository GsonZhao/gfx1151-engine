#!/usr/bin/env python3
"""D0-M5: 并发卡顿基线（HANDOFF-CONCURRENCY §4-M5）。

会话 A 持续 decode（短 prompt、chain drafter、512 token），3 秒后会话 B
提交 32K prompt。记录 A 的逐 token 间隔（max/p50）与 B 的 TTFT。
预期（当前分时实现，32K chunk ≈ 25 s）：A 的最大间隔 ≈ B 的整个 prefill。

用法: python3 tools/c0_stall.py   （引擎已在 127.0.0.1:8732 以 PARALLEL=2 起好）
"""
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qwentok import Tokenizer
from a5_ab import prompts

PORT = 8732


class Conn:
    def __init__(self, timeout=1800):
        self.sock = socket.create_connection(('127.0.0.1', PORT), timeout=timeout)
        self.f = self.sock.makefile('rb')
        self.req = 0

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def line(self):
        v = self.f.readline().decode().split()
        if not v:
            raise RuntimeError('engine disconnected')
        return v

    def send_gen(self, ids, n, drafter):
        self.req += 1
        fields = ['GEN', self.req, n, 0, len(ids), *ids, drafter]
        self.sock.sendall((' '.join(map(str, fields)) + '\n').encode())
        return self.req

    def read_gen(self, on_tok=None):
        out = []
        while True:
            v = self.line()
            if v[0] == 'T':
                out.append(int(v[2]))
                if on_tok:
                    on_tok(len(out))
            elif v[0] == 'D':
                return dict(tokens=out, reason=v[2], done=' '.join(v))


def main():
    tk = Tokenizer()
    _, o32, natural, _, _ = prompts(tk)

    a_times = []
    a_err = []

    def run_a():
        try:
            c = Conn()
            c.send_gen(natural, 512, 4)  # chain drafter（API 默认）
            c.read_gen(on_tok=lambda i: a_times.append(time.monotonic()))
            c.close()
        except Exception as e:
            a_err.append(e)

    ta = threading.Thread(target=run_a, daemon=True)
    t0 = time.monotonic()
    ta.start()

    time.sleep(3)  # 等 A 进入稳定 decode

    c2 = Conn()
    tb0 = time.monotonic()
    c2.send_gen(o32, 8, 0)  # B：32K prompt，串行 decode 只要 TTFT
    b_ttft = []

    def on_b(i):
        if i == 1:
            b_ttft.append(time.monotonic() - tb0)

    c2.read_gen(on_tok=on_b)
    c2.close()
    ta.join(timeout=600)

    if a_err:
        print('A 会话异常:', a_err[0])
        sys.exit(1)
    gaps = [b - a for a, b in zip(a_times, a_times[1:])]
    gaps.sort()
    p50 = gaps[len(gaps) // 2]
    print(f'A tokens={len(a_times)} 间隔 p50={p50 * 1000:.0f} ms '
          f'max={gaps[-1] * 1000:.0f} ms')
    print(f'A 首个 token 延迟={(a_times[0] - t0) * 1000:.0f} ms')
    if b_ttft:
        print(f'B 32K prompt TTFT={b_ttft[0]:.1f} s')
    print('结论行: A_max_gap=%.1f s  B_TTFT=%.1f s' %
          (gaps[-1], b_ttft[0] if b_ttft else -1))


if __name__ == '__main__':
    main()
