from re_arc.dsl import *
def generate_253bf280() -> float:
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (3, 30)
    colopts = remove(3, interval(0, 10, 1))
    c = canvas(bgc, (20, 19))
    inds = totuple(asindices(c))
    card_bounds = (0, max(1, (20 * 19) // 4))
    gi = fill(c, fgcol, s)
    return gi