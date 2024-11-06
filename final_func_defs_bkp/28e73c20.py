from re_arc.dsl import *
def generate_28e73c20() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (3,))
    sp = (0, 29 - 1)
    gi = canvas(-1, (20, 29))
    inds = asindices(gi)
    obj = {(choice(ccols), ij) for ij in inds}
    gi = paint(gi, obj)
    lw = 29
    lh = 20
    ld = 20
    isverti = False
    while ld > 0:
        lw -= 1
        lh -= 1
        ep = add(sp, multiply((1, 0), ld - 1))
        ln = connect(sp, ep)
        gi = fill(gi, 3, ln)
        if isverti:
            ld = lh
        else:
            ld = lw
        isverti = not isverti
        sp = ep
    gi = dmirror(dmirror(gi)[1:])
    return gi