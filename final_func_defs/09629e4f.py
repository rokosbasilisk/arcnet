from re_arc.dsl import *
def generate_09629e4f() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    nrows, ncolumns = 3, 5
    remcols = remove(bgc, cols)
    remcols = remove(barcol, remcols)
    c = canvas(bgc, (3, 5))
    inds = totuple(asindices(c))
    fullh, fullw = 3 * nrows + nrows - 1, 5 * ncolumns + 1
    gi = canvas(barcol, (fullh, fullw))
    locs = totuple(product(interval(0, fullh, 3 + 1), interval(0, fullw, 5 + 1)))
    remlocs = remove(trgloc, locs)
    colssf = sample(remcols, ncols)
    colsss = remove(choice(colssf), colssf)
    trgssf = sample(inds, ncols - 1)
    gi = fill(gi, bgc, shift(inds, trgloc))
    for ij, cl in zip(trgssf, colsss):
        gi = fill(gi, cl, {add(trgloc, ij)})
    for rl in remlocs:
        trgss = sample(inds, ncols)
        tmpg = tuple(e for e in c)
        for ij, cl in zip(trgss, colssf):
            tmpg = fill(tmpg, cl, {ij})
        gi = paint(gi, shift(asobject(tmpg), rl))
    return gi