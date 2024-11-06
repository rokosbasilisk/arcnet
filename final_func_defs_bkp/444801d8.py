from re_arc.dsl import *
def generate_444801d8() -> None:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (15, 14))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        ow = 5
        bx = box({(1, 0), (oh - 1, 4)}) - {(1, 2)}
        fullobj = backdrop({(0, 0), (oh - 1, 4)})
        cands = backdrop(bx) - bx
        dcol, bxcol = sample(ccols, 2)
        inobj = recolor(bxcol, bx) | recolor(dcol, {dot})
        outobj = recolor(bxcol, bx) | recolor(dcol, fullobj - bx)
        if choice((True, False)):
            inobj = shift(hmirror(inobj), UP)
            outobj = hmirror(outobj)
        cands = sfilter(inds, lambda ij: ij[0] <= 15 - oh and ij[1] <= 14 - ow)
        if len(cands) == 0:
            continue
        outplcd = shift(outobj, loc)
        outplcdi = toindices(outplcd)
        if outplcdi.issubset(inds):
            succ += 1
            inplcd = shift(inobj, loc)
            inds = (inds - outplcdi) - outbox(inplcd)
            gi = paint(gi, inplcd)
    return gi