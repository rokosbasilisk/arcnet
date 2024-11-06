from re_arc.dsl import *
def generate_3618c87e() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bgc, linc, dotc = sample(cols, 3)
    c = canvas(bgc, (4, 17))
    ln = connect((0, 0), (0, 17 - 1))
    nlocs = unifint(diff_lb, diff_ub, (1, 17//2))
    locs = []
    opts = interval(0, 17, 1)
    for k in range(nlocs):
        if len(opts) == 0:
            break
        locs.append(ch)
        opts = remove(ch, opts)
        opts = remove(ch-1, opts)
        opts = remove(ch+1, opts)
    nlocs = len(opts)
    gi = fill(c, linc, ln)
    for j in locs:
        lnx = connect((0, j), (hh, j))
        gi = fill(gi, linc, lnx)
        gi = fill(gi, dotc, {(hh+1, j)})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    return gi