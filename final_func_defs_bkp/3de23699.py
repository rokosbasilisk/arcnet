from re_arc.dsl import *
def generate_3de23699() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bgc = choice(cols)
    c = canvas(bgc, (30, 27))
    remcols = remove(bgc, cols)
    remcols = remove(ccol, remcols)
    tmpo = frozenset({(loci, locj), (loci + hi - 1, locj + wi - 1)})
    cnds = totuple(backdrop(inbox(tmpo)))
    mp = len(cnds) // 2
    ss = sample(cnds, ncnds)
    gi = fill(c, ccol, corners(tmpo))
    gi = fill(gi, ncol, ss)
    return gi