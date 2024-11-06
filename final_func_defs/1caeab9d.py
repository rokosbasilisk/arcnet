from re_arc.dsl import *
def generate_1caeab9d() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1,))
    bb = asindices(canvas(-1, (oh, ow)))
    obj = {sp}
    bb = remove(sp, bb)
    for k in range(ncells):
        obj.add(choice(totuple((bb - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    itv = interval(0, 23, 1)
    objp = shift(obj, (loci, locj))
    remcols = remove(bgc, cols)
    c = canvas(bgc, (26, 23))
    gi = fill(c, 1, objp)
    for k in range(numo):
        cands = sfilter(itv, lambda j: set(interval(j, j + ow, 1)).issubset(set(itv)))
        if len(cands) == 0:
            break
        remcols = remove(col, remcols)
        gi = fill(gi, col, shift(obj, (randint(0, 26 - oh), locj)))
        itv = difference(itv, interval(locj, locj + ow, 1))
    return gi