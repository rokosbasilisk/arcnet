from re_arc.dsl import *
def generate_484b58aa() -> tuple:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    pinds = asindices(canvas(-1, (hp, wp)))
    remcols = remove(noisec, cols)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    locs = set()
    numpatches = unifint(diff_lb, diff_ub, (1, (16 * 20) // 20))
    gi = tuple(e for e in canvas(-1, (16, 20)))
    return gi