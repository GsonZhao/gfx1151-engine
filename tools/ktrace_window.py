#!/usr/bin/env python3
"""decode 窗口内按 (kernel, blocks, grid_y) 聚合，除以轮数/token 数。
用法: python3 ktrace_window.py <ktrace目录或db> <除数> [--after-first REGEX] [--top N]
  除数: 投机用日志里的 rounds=，串行用 GEN 的 token 数
  --after-first: 窗口起点 = 第一个名字匹配 REGEX 的 kernel（默认 k_argmax，即 prefill 结束后）"""
import glob, os, re, sqlite3, sys
a = sys.argv[1:]
path, div = a[0], float(a[1])
rx = a[a.index("--after-first") + 1] if "--after-first" in a else "k_argmax"
top = int(a[a.index("--top") + 1]) if "--top" in a else 40
db = path if path.endswith(".db") else glob.glob(os.path.join(path, "**", "*.db"), recursive=True)[0]
c = sqlite3.connect(db)
rows = c.execute("select name,start,end,duration,grid_x,grid_y,workgroup_x from kernels order by start").fetchall()
t0 = next(r[1] for r in rows if re.search(rx, r[0]))
w = [r for r in rows if r[1] >= t0]
busy = sum(r[3] for r in w); wall = w[-1][2] - t0
small = [r for r in w if r[4] // max(r[6], 1) <= 20 and r[5] == 1]
print("窗口 %.0f ms, 除数 %g: busy %.2f ms/单位, wall %.2f ms/单位, launch %.0f/单位" %
      (wall / 1e6, div, busy / 1e6 / div, wall / 1e6 / div, len(w) / div))
print("小 grid(<=20 块): %.2f ms/单位, %.0f 次/单位" % (sum(r[3] for r in small) / 1e6 / div, len(small) / div))
agg = {}
for n, s, e, d, gx, gy, wx in w:
    n = re.sub(r"\(.*$", "", n).replace("void ", "")[:70]
    k = (n, gx // max(wx, 1), gy)
    v = agg.setdefault(k, [0, 0]); v[0] += d; v[1] += 1
print("%8s %7s %8s %6s %5s  %s" % ("ms/单位", "次/单位", "avg us", "blocks", "gy", "kernel"))
for (n, b, gy), (d, cnt) in sorted(agg.items(), key=lambda x: -x[1][0])[:top]:
    print("%8.2f %7.1f %8.1f %6d %5d  %s" % (d / 1e6 / div, cnt / div, d / 1e3 / cnt, b, gy, n))
