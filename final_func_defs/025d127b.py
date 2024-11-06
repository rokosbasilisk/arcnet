from re_arc.dsl import *
def generate_025d127b() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (19, 26))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= 19 - oh and ij[1] <= 26 - ow)
        if len(cands) == 0:
            continue
        topl = connect((0, 0), (0, ow - 1))
        leftl = connect((1, 0), (oh - 2, oh - 3))
        rightl = connect((1, ow), (oh - 2, ow + oh - 3))
        botl = connect((oh - 1, oh - 2), (oh - 1, oh - 3 + ow))
        inobj = topl | leftl | rightl | botl
        outobj = shift(topl, (0, 1)) | botl | shift(leftl, (0, 1)) | connect((1, ow+1), (oh - 3, ow + oh - 3)) | {(oh - 2, ow + oh - 3)}
        outobj = sfilter(outobj, lambda ij: ij[1] <= rightmost(inobj))
        fullobj = inobj | outobj
        inobj = shift(inobj, loc)
        outobj = shift(outobj, loc)
        fullobj = shift(fullobj, loc)
        if fullobj.issubset(inds):
            inds = (inds - fullobj) - mapply(neighbors, fullobj)
            succ += 1
            col = choice(ccols)
            gi = fill(gi, col, inobj)
    return gi