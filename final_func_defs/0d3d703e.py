from re_arc.dsl import *
def generate_0d3d703e() -> float:
    diff_lb = 0
    diff_ub = 1
    incols = (1, 2, 3, 4, 5, 6, 8, 9)
    k = len(incols)
    gi = canvas(-1, (19, 21))
    inds = asindices(gi)
    for ij in inds:
        gi = fill(gi, incols[idx], {ij})
    return gi