from re_arc.dsl import *
def generate_1fad071e() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = remove(1, interval(0, 10, 1))
    bgc, otherc = sample(cols, 2)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    bcount = 0
    gi = canvas(bgc, (17, 15))
    inds = asindices(gi)
    ofcfrbinds = {1: set(), otherc: set()}
    while succ < nobjs and tr < maxtr:
        tr += 1
        if bcount < nbl:
            oh, ow = 2, 2
        else:
            while col == 1 and oh == ow == 2:
        bd = backdrop(frozenset({(0, 0), (oh - 1, ow - 1)}))
        cands = sfilter(inds, lambda ij: ij[0] <= 17 - oh and ij[1] <= 15 - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bd = shift(bd, loc)
        if bd.issubset(inds) and len(mapply(dneighbors, bd) & ofcfrbinds[col]) == 0:
            succ += 1
            inds = inds - bd
            ofcfrbinds[col] = ofcfrbinds[col] | mapply(dneighbors, bd) | bd
            gi = fill(gi, col, bd)
            if col == 1 and oh == ow == 2:
                bcount += 1
    return gi