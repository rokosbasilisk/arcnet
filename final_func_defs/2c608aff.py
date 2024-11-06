from re_arc.dsl import *
def generate_2c608aff() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    remcols = remove(bgc, cols)
    remcols = remove(ccol, remcols)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (23, 14))
    gi = fill(gi, ccol, bd)
    reminds = totuple(asindices(gi) - backdrop(outbox(bd)))
    noiseb = max(1, len(reminds) // 4)
    gi = fill(gi, dcol, noise)
    return gi