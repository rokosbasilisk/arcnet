from re_arc.dsl import *
def generate_1e0a9b12() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    ff = chain(dmirror, lbind(apply, rbind(order, identity)), dmirror)
    while True:
        gi = canvas(bgc, (12, 23))
        remcols = remove(bgc, cols)
        inds = totuple(connect(ORIGIN, (12 - 1, 0)))
        for c, l in zip(scols, slocs):
            sel = sample(inds, nc2)
            gi = fill(gi, c, shift(sel, tojvec(l)))
        if colorcount(gi, bgc) > argmax(remove(bgc, palette(gi)), lbind(colorcount, gi)):
            break
    return gi