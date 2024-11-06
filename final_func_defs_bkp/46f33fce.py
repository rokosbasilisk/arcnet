from re_arc.dsl import *
def generate_46f33fce() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    inds = totuple(asindices(gi))
    f = lambda cij: (cij[0], double(cij[1]))
    return gi