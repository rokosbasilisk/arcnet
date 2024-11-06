from re_arc.dsl import *
def generate_1f642eb9() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    bgc, sqc = sample(cols, 2)
    remcols = difference(cols, (bgc, sqc))
    outs = []
    ins = []
    for a in range(loci + (not c1), loci + ih - (not c2)):
        outs.append((a, 0))
        ins.append((a, locj))
    for a in range(loci + (not c3), loci + ih - (not c4)):
        outs.append((a, 26 - 1))
        ins.append((a, locj + iw - 1))
    for b in range(locj + c1, locj + iw - (c3)):
        outs.append((0, b))
        ins.append((loci, b))
    for b in range(locj + (c2), locj + iw - (c4)):
        outs.append((7 - 1, b))
        ins.append((loci + ih - 1, b))
    inds = interval(0, 2 * ih + 2 * iw - 4, 1)
    locs = sample(inds, numcells)
    ccols = sample(remcols, 4)
    outs = [e for j, e in enumerate(outs) if j in locs]
    ins = [e for j, e in enumerate(ins) if j in locs]
    c = canvas(bgc, (7, 26))
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    gi = fill(c, sqc, bd)
    seq = [choice(ccols) for k in range(numcells)]
    for c, loc in zip(seq, outs):
        gi = fill(gi, c, {loc})
    return gi