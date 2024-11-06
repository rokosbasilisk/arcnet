from re_arc.dsl import *
def generate_496994bd() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    remcols = remove(bgc, cols)
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (9, 10))
    bx = asindices(canv)
    obj = {
        (choice(remcols), choice(totuple(sfilter(bx, lambda ij: ij[0] < 9//2)))),
        (choice(remcols), choice(totuple(sfilter(bx, lambda ij: ij[0] > 9//2))))
    }
    for kk in range(nc - 2):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gix = paint(canv, obj)
    gix = apply(rbind(order, matcher(identity, bgc)), gix)
    flag = choice((True, False))
    gi = hconcat(gix, canv if flag else hconcat(canvas(bgc, (9, 1)), canv))
    if choice((True, False)):
        gi = vmirror(gi)
    if choice((True, False)):
        gi = hmirror(gi)
    return gi