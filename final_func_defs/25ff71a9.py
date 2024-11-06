from re_arc.dsl import *
def generate_25ff71a9() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    nc = unifint(diff_lb, diff_ub, (1, (12 * 28) // 2 - 1))
    remcols = remove(bgc, cols)
    c = canvas(bgc, (12, 28))
    bounds = asindices(c)
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc-1):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, 1, plcd)
    return gi