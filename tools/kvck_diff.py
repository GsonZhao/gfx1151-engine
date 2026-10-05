import struct, glob, sys

D = '/home/mark/Workspace/gfx-1151-kvsnap/data/kvsnap'

# c_<key>.kvc: ChHdr = magic[8] + key,parent,sum (3Q) + nl,kvb,has_ik,look (4i) + toks[256] (256i)
pages = {}
for f in glob.glob(D + '/c_*.kvc'):
    with open(f, 'rb') as fh:
        b = fh.read(48 + 1024)
    if b[:6] != b'GDKVC1':
        continue
    key = struct.unpack('<Q', b[8:16])[0]
    pages[key] = struct.unpack('<256i', b[48:48 + 1024])
print('pages indexed:', len(pages), file=sys.stderr)

# k_<n>_<id>.kck: CkHdr = magic[8] + seed,base,id,sum (4Q) + 10i + 6Q  => 128 bytes
# body: nc * 16 (page key + mtp section key), then nt * 4 tail tokens
def ck_tokens(fn):
    with open(fn, 'rb') as fh:
        b = fh.read()
    assert b[:6] == b'GDKCK1', fn
    n, nc = struct.unpack('<2i', b[40:48])
    off = 128
    keys = struct.unpack('<%dQ' % nc, b[off:off + nc * 8])
    toks = []
    for k in keys:
        toks.extend(pages[k])
    off += nc * 16  # page keys, then mtp section keys
    nt = n - nc * 256
    toks.extend(struct.unpack('<%di' % nt, b[off:off + nt * 4]))
    assert len(toks) == n, (fn, len(toks), n)
    return toks

a = ck_tokens(D + '/k_79099_b7c41e774b557e59.kck')   # REQ 1: generated stream
b = ck_tokens(D + '/k_79080_af3a618d9584eb30.kck')   # REQ 3: retokenized prompt cut
c = ck_tokens(D + '/k_79098_e2da937518fa5f1c.kck')   # REQ 3: second cut

print('len a=%d b=%d c=%d' % (len(a), len(b), len(c)))
n = min(len(a), len(b))
i = 0
while i < n and a[i] == b[i]:
    i += 1
print('first divergence at token index %d (page %d, offset %d)' % (i, i // 256, i % 256))
lo = max(0, i - 8)
print('a[%d:%d] =' % (lo, i + 12), a[lo:i + 12])
print('b[%d:%d] =' % (lo, i + 12), b[lo:i + 12])
# do they re-sync later? try aligning b against a with small shifts
for shift in range(-4, 5):
    j = i
    while j < n and a[j] == b[j + shift]:
        j += 1
    print('shift %+d: match from %d to %d' % (shift, i, j))
# b vs c consistency check (both from REQ 3's prompt)
k = 0
while k < min(len(b), len(c)) and b[k] == c[k]:
    k += 1
print('b vs c common prefix:', k)
