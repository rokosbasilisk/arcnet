from re_arc.dsl import *
def generate_44d8ac46() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (14, 23))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        tr += 1
        if len(remcols) == 0 or len(indss) == 0:
            break
        subs = totuple(sfilter(indss, lambda ij: ij[0] < 14 - oh and ij[1] < 23 - ow))
        if len(subs) == 0:
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        if bd.issubset(indss):
            ensuresq = choice((True, False))
            if ensuresq:
                inpart = backdrop({(loci + iloci, locj + ilocj), (loci + iloci + dim - 1, locj + ilocj + dim - 1)})
            else:
                cnds = backdrop(inbox(bd))
                ch = choice(totuple(cnds))
                inpart = {ch}
                for k in range(kk - 1):
                    inpart.add(choice(totuple((cnds - inpart) & mapply(dneighbors, inpart))))
            inpart = frozenset(inpart)
            hi, wi = shape(inpart)
            if hi == wi and len(inpart) == hi * wi:
                incol = 2
            else:
                incol = bgc
            gi = fill(gi, col, bd)
            gi = fill(gi, bgc, inpart)
            succ += 1
            indss = (indss - bd) - outbox(bd)
    return gi