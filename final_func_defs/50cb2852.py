from re_arc.dsl import *
def generate_50cb2852() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (14, 13))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        subs = totuple(sfilter(indss, lambda ij: ij[0] < 14 - oh and ij[1] < 13 - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            succ += 1
            indss = indss - bd
        tr += 1
    return gi