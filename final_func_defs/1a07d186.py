from re_arc.dsl import *
def generate_1a07d186() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    remcols = difference(remcols, linecols)
    locopts = interval(0, 25, 1)
    locs = []
    for k in range(nlines):
        if len(locopts) == 0:
            break
        locopts = difference(locopts, interval(loc - 2, loc + 3, 1))
        locs.append(loc)
    locs = sorted(locs)
    gi = canvas(bgc, (27, 25))
    for loc, col in zip(locs, linecols):
        gi = fill(gi, col, connect((0, loc), (27 - 1, loc)))
    dotlocopts = difference(interval(0, 25, 1), locs)
    for ii in ilocs:
        for dotlocj, col in zip(dotlocs, dotcols):
            gi = fill(gi, col, {(ii, dotlocj)})
    if choice((True, False)):
        gi = dmirror(gi)
    return gi