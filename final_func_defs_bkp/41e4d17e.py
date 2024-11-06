from re_arc.dsl import *
def generate_41e4d17e() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = remove(6, interval(0, 10, 1))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (9, 23))
    inds = asindices(gi)
    bx = box(frozenset({(0, 0), (4, 4)}))
    bd = backdrop(bx)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    while succ < num and tr < maxtrials:
        bxs = shift(bx, loc)
        if bxs.issubset(set(inds)):
            gi = fill(gi, fgc, bxs)
            cen = center(bxs)
            frns = hfrontier(cen) | vfrontier(cen)
            kep = frns & ofcolor(gi, bgc)
            gi = fill(gi, 6, kep)
            inds = difference(inds, shift(bd, loc))
            succ += 1
        tr += 1
    return gi