from re_arc.dsl import *
def generate_1f85a75f() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {sp}
    cands = remove(sp, bounds)
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    bgc, objc = sample(cols, 2)
    remcols = remove(bgc, remove(objc, cols))
    gi = canvas(bgc, (15, 15))
    obj = shift(obj, (loci, locj))
    gi = fill(gi, objc, obj)
    inds = asindices(gi)
    noisecells = sample(totuple(inds - backdrop(obj)), nnoise)
    noiseobj = frozenset({(choice(ccols), ij) for ij in noisecells})
    gi = paint(gi, noiseobj)
    return gi