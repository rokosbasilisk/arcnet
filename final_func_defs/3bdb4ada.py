from re_arc.dsl import *
def generate_3bdb4ada() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (27, 28))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        if choice((True, False)):
        else:
        subs = totuple(sfilter(indss, lambda ij: ij[0] < 27 - oh and ij[1] < 28 - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            if oh == 3:
                ln = {(loci + 1, j) for j in range(locj+1, locj+ow, 2)}
            else:
                ln = {(j, locj + 1) for j in range(loci+1, loci+oh, 2)}
            gi = fill(gi, bgc, ln)
            succ += 1
            indss = indss - bd
        tr += 1
    return gi