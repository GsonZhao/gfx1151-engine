#!/usr/bin/env python3
"""Windows 启动器图标：PNG → src/launch_win.ico + src/launch_win_icon.inc

用法（仓库根目录）：
    python3 tools/win_icon.py                      # 默认读 src/launch_win_icon.png
    python3 tools/win_icon.py path/to/logo.png     # 换图标

- launch_win.ico：多尺寸 ICO（32 位 BMP 条目 + AND 掩码，兼容性最好），build_win.sh 有 llvm-rc
  时编进 exe 资源，资源管理器里的文件图标用它。
- launch_win_icon.inc：同一个 ICO 的字节数组，start_win.exe 运行时自己解析，
  托盘图标用它（不依赖资源编译器，没有 llvm-rc 也有托盘图标）。
源图最好是正方形、透明背景；只生成不大于源图的尺寸（不放大）。
"""
import os
import struct
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]


def bmp_entry(im: Image.Image, s: int) -> bytes:
    """32 位 BGRA DIB（自下而上）+ 1bpp AND 掩码——标准 ICO 图像条目。
    （Pillow 的 bitmap_format="bmp" 不写 AND 掩码，Windows 按标准会越界读。）"""
    px = im.resize((s, s), Image.LANCZOS).load()
    xor = bytearray()
    mask_row = ((s + 31) // 32) * 4
    mask = bytearray()
    for y in range(s - 1, -1, -1):
        row = bytearray(mask_row)
        for x in range(s):
            r, g, b, a = px[x, y]
            xor += bytes((b, g, r, a))
            if a == 0:
                row[x >> 3] |= 0x80 >> (x & 7)
        mask += row
    hdr = struct.pack("<IiiHHIIiiII", 40, s, 2 * s, 1, 32, 0, len(xor) + len(mask), 0, 0, 0, 0)
    return hdr + bytes(xor) + bytes(mask)


def build_ico(im: Image.Image, sizes: list) -> bytes:
    imgs = [bmp_entry(im, s) for s in sizes]
    out = bytearray(struct.pack("<HHH", 0, 1, len(imgs)))
    off = 6 + 16 * len(imgs)
    for s, b in zip(sizes, imgs):
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(b), off)
        off += len(b)
    for b in imgs:
        out += b
    return bytes(out)


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "launch_win_icon.png")
    im = Image.open(src).convert("RGBA")
    if im.width != im.height:
        side = max(im.size)
        sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        sq.alpha_composite(im, ((side - im.width) // 2, (side - im.height) // 2))
        im = sq
    sizes = [(s, s) for s in SIZES if s <= im.width]
    ico = os.path.join(ROOT, "src", "launch_win.ico")
    data = build_ico(im, [s[0] for s in sizes])
    with open(ico, "wb") as f:
        f.write(data)

    lines = [
        "// 由 tools/win_icon.py 生成，勿手改。内容 = src/launch_win.ico 的字节。",
        f"// 尺寸：{', '.join(str(s[0]) for s in sizes)}",
        f"static const unsigned char kIconIco[{len(data)}] = {{",
    ]
    for i in range(0, len(data), 24):
        lines.append("    " + ",".join(f"0x{b:02x}" for b in data[i:i + 24]) + ",")
    lines.append("};")
    inc = os.path.join(ROOT, "src", "launch_win_icon.inc")
    with open(inc, "wb") as f:
        f.write(("\r\n".join(lines) + "\r\n").encode("utf-8"))  # 与 launch_win.cpp 一样用 CRLF
    print(f"{ico}: {len(data)} 字节，尺寸 {[s[0] for s in sizes]}")
    print(f"{inc}: 已更新")


if __name__ == "__main__":
    main()
