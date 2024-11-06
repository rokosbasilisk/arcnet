from re_arc.dsl import *
def generate_10fcaaa3() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    remcols = remove(bgc, cols)
    c = canvas(bgc, (9, 10))
    inds = asindices(c)
    locs = frozenset(sample(totuple(inds), ncells))
    obj = frozenset({(choice(ccols), ij) for ij in locs})
    gi = paint(c, obj)
    return gi