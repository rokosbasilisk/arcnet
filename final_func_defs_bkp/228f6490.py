from re_arc.dsl import *
def generate_228f6490() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    succ = 0
    tr = 0
    maxtr = 5 * nsq
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (28, 22))
    inds = asindices(gi)
    forbidden = []
    while tr < maxtr and succ < nsq:
        tr += 1
        bd = asindices(canvas(-1, (oh, ow)))
        bounds = shift(asindices(canvas(-1, (oh-2, ow-2))), (1, 1))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        sqcands = sfilter(inds, lambda ij: ij[0] <= 28 - oh and ij[1] <= 22 - ow)
        if len(sqcands) == 0:
            continue
        bdplcd = shift(bd, loc)
        if bdplcd.issubset(inds):
            tmpinds = inds - bdplcd
            inobjn = normalize(obj)
            oh, ow = shape(obj)
            inobjcands = sfilter(inds, lambda ij: ij[0] <= 28 - oh and ij[1] <= 22 - ow)
            if len(inobjcands) == 0:
                continue
            inobjplcd = shift(inobjn, loc2)
            bdnorm = bd - obj
            if inobjplcd.issubset(tmpinds) and bdnorm not in forbidden and inobjn not in forbidden:
                forbidden.append(bdnorm)
                forbidden.append(inobjn)
                succ += 1
                inds = (inds - (bdplcd | inobjplcd)) - mapply(dneighbors, inobjplcd)
                oplcd = shift(obj, loc)
                gi = fill(gi, sqc, bdplcd - oplcd)
                gi = fill(gi, col, inobjplcd)
    succ = 0
    tr = 0
    maxtr = 10 * nremobjs
    while tr < maxtr and succ < nremobjs:
        tr += 1
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        if obj in forbidden:
            continue
        cands = sfilter(inds, lambda ij: ij[0] <= 28 - oh and ij[1] <= 22 - ow)
        if len(cands) == 0:
            continue
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            gi = fill(gi, col, plcd)
    return gi