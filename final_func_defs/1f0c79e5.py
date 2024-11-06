from re_arc.dsl import *
def generate_1f0c79e5() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    bgc, objc = sample(cols, 2)
    gi = canvas(bgc, (21, 13))
    inds = asindices(gi)
    obj = ((0, 0), (0, 1), (1, 0), (1, 1))
    for k in range(nobjs):
        cands = sfilter(inds, lambda ij: shift(set(obj), ij).issubset(inds))
        if len(cands) == 0:
            break
        plcd = shift(obj, loc)
        gi = fill(gi, objc, plcd)
        gi = fill(gi, 2, reds)
        for idx in reds:
            direc = decrement(multiply(2, add(idx, invert(loc))))
            gi = fill(gi, objc, mapply(rbind(shoot, direc), frozenset(plcd)))
        inds = (inds - plcd) - mapply(dneighbors, set(plcd))
    return gi