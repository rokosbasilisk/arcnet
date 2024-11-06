from re_arc.dsl import *
def generate_4258a5f9() -> any:
    diff_lb = 0
    diff_ub = 1
    colopts = remove(1, interval(0, 10, 1))
    remcols = remove(bgc, colopts)
    gi = canvas(bgc, (6, 29))
    mp = ((6 * 29) // 2) if (6 * 29) % 2 == 1 else ((6 * 29) // 2 - 1)
    inds = totuple(asindices(gi))
    gi = fill(gi, 9, dots)
    return gi