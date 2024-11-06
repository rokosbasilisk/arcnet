from re_arc.dsl import *
def generate_22233c11() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    remcols = remove(bgc, cols)
    inds = asindices(gi)
    fullinds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        if len(inds) == 0:
            break
        tr += 1
        fulld = 4 * od
        g = canvas(bgc, (4, 4))
        g = fill(g, 8, {(0, 3), (3, 0)})
        g = fill(g, col, {(1, 1), (2, 2)})
        if choice((True, False)):
            g = hmirror(g)
        g = upscale(g, od)
        inobj = recolor(col, ofcolor(g, col))
        outobj = inobj | recolor(8, ofcolor(g, 8))
        loc = choice(totuple(inds))
        outobj = shift(outobj, loc)
        inobj = shift(inobj, loc)
        outobji = toindices(outobj)
        if toindices(inobj).issubset(inds) and (outobji & fullinds).issubset(inds):
            succ += 1
            inds = (inds - outobji) - mapply(neighbors, outobji)
    return gi