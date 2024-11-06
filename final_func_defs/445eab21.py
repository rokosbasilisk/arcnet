from re_arc.dsl import *
def generate_445eab21() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (26, 14))
    indss = asindices(gi)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    bigcol, area = 0, 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        if oh * ow == area:
            continue
        subs = totuple(sfilter(indss, lambda ij: ij[0] < 26 - oh and ij[1] < 14 - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, box(bd))
            succ += 1
            indss = indss - bd
            if oh * ow > area:
                bigcol, area = col, oh * ow
        tr += 1
    return gi