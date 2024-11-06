from re_arc.dsl import *
def generate_3631a71a() -> tuple:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bgc, patchcol = sample(cols, 2)
    remcols = difference(cols, (bgc, patchcol))
    c = canvas(bgc, (10, 10))
    inds = sfilter(asindices(c), lambda ij: ij[0] >= ij[1])
    cells = set(sample(totuple(inds), ncells))
    obj = {(choice(ccols), ij) for ij in cells}
    c = paint(dmirror(paint(c, obj)), obj)
    c = hconcat(c, vmirror(c))
    c = vconcat(c, hmirror(c))
    cutoff = 2
    gi = tuple(e for e in c[:-cutoff])
    forbidden = asindices(canvas(-1, (cutoff, cutoff)))
    dmirrareaL = shift(asindices(canvas(-1, (10*2-2*cutoff, cutoff))), (cutoff, 0))
    dmirrareaT = shift(asindices(canvas(-1, (cutoff, 2*10-2*cutoff))), (0, cutoff))
    inds1 = sfilter(asindices(gi), lambda ij: cutoff <= ij[0] < 10 and cutoff <= ij[1] < 10 and ij[0] >= ij[1])
    inds2 = dmirror(inds1)
    inds3 = shift(hmirror(inds1), (10-cutoff, 0))
    inds4 = shift(hmirror(inds2), (10-cutoff, 0))
    inds5 = shift(vmirror(inds1), (0, 10-cutoff))
    inds6 = shift(vmirror(inds2), (0, 10-cutoff))
    inds7 = shift(hmirror(vmirror(inds1)), (10-cutoff, 10-cutoff))
    inds8 = shift(hmirror(vmirror(inds2)), (10-cutoff, 10-cutoff))
    f1 = identity
    f2 = dmirror
    f3 = lambda x: hmirror(shift(x, invert((10-cutoff, 0))))
    f4 = lambda x: dmirror(hmirror(shift(x, invert((10-cutoff, 0)))))
    f5 = lambda x: vmirror(shift(x, invert((0, 10-cutoff))))
    f6 = lambda x: dmirror(vmirror(shift(x, invert((0, 10-cutoff)))))
    f7 = lambda x: vmirror(hmirror(shift(x, invert((10-cutoff, 10-cutoff)))))
    f8 = lambda x: dmirror(vmirror(hmirror(shift(x, invert((10-cutoff, 10-cutoff))))))
    indsarr = [inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8]
    farr = [f1, f2, f3, f4, f5, f6, f7, f8]
    succ = 0
    tr = 0
    maxtr = 10 * ndist
    fullh, fullw = shape(gi)
    while succ < ndist and tr < maxtr:
        tr += 1
        bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        isleft = set()
        gi2 = fill(gi, patchcol, bd)
        if patchcol in palette(toobject(forbidden, gi2)):
            continue
        oo1 = toindices(sfilter(toobject(dmirrareaL, gi2), lambda cij: cij[0] != patchcol))
        oo2 = toindices(sfilter(toobject(dmirrareaT, gi2), lambda cij: cij[0] != patchcol))
        oo2 = frozenset({(ij[1], ij[0]) for ij in oo2})
        if oo1 | oo2 != dmirrareaL:
            continue
        for ii, ff in zip(indsarr, farr):
            oo = toobject(ii, gi2)
            rem = toindices(sfilter(oo, lambda cij: cij[0] != patchcol))
            if len(rem) > 0:
                isleft = isleft | ff(rem)
        if isleft != inds1:
            continue
        succ += 1
        gi = gi2
    return gi