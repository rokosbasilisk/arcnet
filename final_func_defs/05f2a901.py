from re_arc.dsl import *
def generate_05f2a901() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bb = asindices(canvas(-1, (objh, objw)))
    obj = {sp}
    bb = remove(sp, bb)
    for k in range(ncells - 1):
        obj.add(choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    if height(obj) * width(obj) == len(obj):
        obj = remove(choice(totuple(obj)), obj)
    obj = normalize(obj)
    objh, objw = shape(obj)
    loc = (loci, locj)
    bgc, fgc, destc = sample(cols, 3)
    gi = canvas(bgc, (13, 21))
    obj = shift(obj, loc)
    gi = fill(gi, fgc, obj)
    locsq = (locisq, locjsq)
    sq = backdrop({(locisq, locjsq), (locisq+sqd-1, locjsq+sqd-1)})
    gi = fill(gi, destc, sq)
    while len(obj & sq) == 0:
        obj = shift(obj, (-1, 0))
    obj = shift(obj, (1, 0))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
    return gi