from re_arc.dsl import *
def generate_00d62c1b() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (10, 25))
    succ = 0
    tr = 0
    maxtr = 5 * nblocks
    inds = asindices(gi)
    while succ < nblocks and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= 10 - oh and ij[1] <= 25 - ow)
        if len(cands) == 0:
            continue
        loci, locj = loc
        bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        bx = bx - set(sample(totuple(corners(bx)), randint(0, 4)))
        if bx.issubset(inds) and len(inds - bx) > (10 * 25) // 2 + 1:
            gi = fill(gi, fgc, bx)
            succ += 1
            inds = inds - bx
    maxnnoise = max(0, (10 * 25) // 2 - 1 - colorcount(gi, fgc))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    objs = objects(gi, T, F, F)
    cands = colorfilter(objs, bgc)
    res = mfilter(cands, compose(flip, rbind(bordering, gi)))
    return gi