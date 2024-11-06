from re_arc.dsl import *
def generate_4093f84a() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    loci1, loci2 = sorted(sample(interval(2, 20 - 2, 1), 2))
    bgc, barc, dotc = sample(cols, 3)
    gi = canvas(bgc, (20, 11))
    for ii in range(loci1, loci2 + 1, 1):
        gi = fill(gi, barc, connect((ii, 0), (ii, 11 - 1)))
    opts = interval(0, 11, 1)
    for l1 in locs1:
        k = unifint(diff_lb, diff_ub, (1, loci1 - 1))
        locsx = sample(interval(0, loci1, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l1), locsx))
    for l2 in locs2:
        k = unifint(diff_lb, diff_ub, (1, 20 - loci2 - 2))
        locsx = sample(interval(loci2 + 1, 20, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l2), locsx))
    if choice((True, False)):
        gi = dmirror(gi)
    return gi