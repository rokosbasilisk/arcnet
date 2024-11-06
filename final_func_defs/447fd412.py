from re_arc.dsl import *
def generate_447fd412() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bgc, indic, mainc = sample(cols, 3)
    if oh * ow < 3:
        if choice((True, False)):
        else:
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {choice(totuple(bounds))}
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    objt = totuple(obj)
    kk = len(obj)
    indicobj = set(sample(objt, nindic))
    mainobj = obj - indicobj
    obj = recolor(indic, indicobj) | recolor(mainc, mainobj)
    gi = canvas(bgc, (20, 16))
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(plcd))
    fullinds = asindices(gi)
    tr = 0
    maxtr = 5 * noccs
    succ = 0
    while succ < noccs and tr < maxtr:
        tr += 1
        outobj = upscale(obj, fac)
        inobj = sfilter(outobj, lambda cij: cij[0] == indic)
        hh, ww = shape(outobj)
        cands = sfilter(inds, lambda ij: ij[0] <= 20 - hh and ij[1] <= 16 - ww)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        inobjp = shift(inobj, loc)
        outobjp = shift(outobj, loc)
        outobjp = sfilter(outobjp, lambda cij: cij[1] in fullinds)
        outobjpi = toindices(outobjp)
        if outobjpi.issubset(inds):
            succ += 1
            inds = (inds - outobjpi) - mapply(neighbors, toindices(inobjp))
            gi = paint(gi, inobjp)
    return gi