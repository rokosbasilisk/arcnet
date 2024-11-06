from re_arc.dsl import *
def generate_1e32b0e9() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bgc, linc, fgc = sample(cols, 3)
    fullh = 4 * nh + (nh - 1)
    fullw = 5 * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (4, 5))
    llocs = set()
    for a in range(0, fullh, 4 + 1):
        for b in range(0, fullw, 5 + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    remlocs = remove(srcloc, llocs)
    smallc2 = canvas(bgc, (4 - 2, 5 - 2))
    inds = asindices(smallc2)
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(ncells):
        ij = choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = shift(shp, (1, 1))
    gg = asobject(fill(smallc, fgc, shp))
    gg2 = asobject(fill(smallc, linc, shp))
    gi = paint(c, shift(gg, srcloc))
    for rl in remlocs:
        subobj = sample(totuple(shp), nleft)
        sg2 = asobject(fill(smallc, fgc, subobj))
        gi = paint(gi, shift(sg2, rl))
    return gi