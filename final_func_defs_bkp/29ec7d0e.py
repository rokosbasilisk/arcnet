from re_arc.dsl import *
def generate_29ec7d0e() -> tuple:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    pinds = asindices(canvas(-1, (hp, wp)))
    bgc, noisec = sample(cols, 2)
    remcols = remove(noisec, cols)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    locs = set()
    numpatches = unifint(diff_lb, diff_ub, (1, (18 * 11) // 20))
    gi = tuple(e for e in canvas(bgc, (18, 11)))
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                succ += 1
                gi = gi2
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    return gi