from re_arc.dsl import *
def generate_46442a0e() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    inds = totuple(asindices(gi))
    for col in colsch:
        inds = difference(inds, chos)
    return gi