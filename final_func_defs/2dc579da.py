from re_arc.dsl import *
def generate_2dc579da() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    remcols = remove(linc, remcols)
    gi = canvas(bgc, (22, 11))
    for a in range(loci, loci + lineh):
        gi = fill(gi, linc, connect((a, 0), (a, 11 - 1)))
    for b in range(locj, locj + linew):
        gi = fill(gi, linc, connect((0, b), (22 - 1, b)))
    dot = backdrop(frozenset({(dotloci, dotlocj), (dotloci + doth - 1, dotlocj + dotw - 1)}))
    gi = fill(gi, dotc, dot)
    return gi