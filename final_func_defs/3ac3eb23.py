from re_arc.dsl import *
def generate_3ac3eb23() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    locopts = interval(1, 27 - 1, 1)
    gi = canvas(bgc, (24, 27))
    for k in range(nlocs):
        if len(locopts) == 0:
            break
        locopts = difference(locopts, interval(locj - 2, locj + 3, 1))
        gi = fill(gi, col, {(0, locj)})
    mf = choice((identity, rot90, rot180, rot270))
    gi = mf(gi)
    return gi