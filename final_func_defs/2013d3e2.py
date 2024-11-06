from re_arc.dsl import *
def generate_2013d3e2() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    remcols = remove(bgc, cols)
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (8, 8))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    gi1 = hconcat(gi, rot90(gi))
    gi2 = hconcat(rot270(gi), rot180(gi))
    gi = vconcat(gi1, gi2)
    gio = asobject(gi)
    gic = canvas(bgc, (fullh, fullw))
    gi = paint(gic, shift(gio, (loci, locj)))
    reminds = difference(asindices(gi), ofcolor(gi, bgc))
    return gi