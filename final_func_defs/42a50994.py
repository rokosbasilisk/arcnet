from re_arc.dsl import *
def generate_42a50994() -> any:
    diff_lb = 0
    diff_ub = 1
    colopts = interval(0, 10, 1)
    remcols = remove(bgc, colopts)
    inds = totuple(asindices(canvas(bgc, (2, 25))))
    chosinds = sample(inds, num)
    choscols = sample(remcols, numcols)
    locs = interval(0, len(chosinds), 1)
    choslocs = sample(locs, numcols)
    gi = canvas(bgc, (2, 25))
    for col, endidx in zip(choscols, sorted(choslocs)[::-1]):
        gi = fill(gi, col, chosinds[:endidx])
    return gi