from re_arc.dsl import *
def generate_321b1fc6() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    objh = unifint(diff_lb, diff_ub, (2, 5))
    objw = unifint(diff_lb, diff_ub, (2, 5))
    bounds = asindices(canvas(0, (objh, objw)))
    for j in range(nc):
        shp.add(ij)
    remcols = remove(bgc, cols)
    remcols = remove(dmyc, remcols)
    oh, ow = shape(shp)
    shpp = shift(shp, (loci, locj))
    shppc = frozenset({(choice(colll), ij) for ij in shpp})
    while numcolors(shppc) == 1:
        shppc = frozenset({(choice(colll), ij) for ij in shpp})
    shppcn = normalize(shppc)
    gi = canvas(bgc, (28, 25))
    gi = paint(gi, shppc)
    return gi