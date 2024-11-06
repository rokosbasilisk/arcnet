from re_arc.dsl import *
def generate_150deff5() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 8))
    bo = {(0, 0), (0, 1), (1, 0), (1, 1)}
    ro1 = {(0, 0), (0, 1), (0, 2)}
    ro2 = {(0, 0), (1, 0), (2, 0)}
    boforb = set()
    reforb = set()
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (16, 13))
    inds = asindices(gi)
    needsbgc = []
    for k in range(noccs):
        obj, col = choice(((bo, 8), (choice((ro1, ro2)), 2)))
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= 16 - oh and ij[1] <= 13 - ow and shift(obj, ij).issubset(inds))
        if col == 8:
            cands = sfilter(cands, lambda ij: ij not in boforb)
        else:
            cands = sfilter(cands, lambda ij: ij not in reforb)
        if len(cands) == 0:
            break
        if col == 8:
            boforb.add(add(loc, (-2, 0)))
            boforb.add(add(loc, (2, 0)))
            boforb.add(add(loc, (0, 2)))
            boforb.add(add(loc, (0, -2)))
        if col == 2:
            if obj == ro1:
                reforb.add(add(loc, (0, 3)))
                reforb.add(add(loc, (0, -3)))
            else:
                reforb.add(add(loc, (1, 0)))
                reforb.add(add(loc, (-1, 0)))
        plcd = shift(obj, loc)
        gi = fill(gi, fgc, plcd)
    return gi