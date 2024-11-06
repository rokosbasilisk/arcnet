from re_arc.dsl import *
def generate_39e1d7f9() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bgc, linc, dotc = sample(cols, 3)
    remcols = difference(cols, (bgc, linc, dotc))
    gi = canvas(bgc, (10, 7))
    if 10 == 5:
    if 7 == 5:
    candsss = neighbors((loci, locj))
    pixs = {(loci, locj)}
    for k in range(npix):
        pixs.add(choice(totuple((mapply(dneighbors, pixs) & candsss) - pixs)))
    pixs = totuple(remove((loci, locj), pixs))
    obj = {(choice(ccols), ij) for ij in pixs}
    gi = fill(gi, dotc, {(loci, locj)})
    gi = paint(gi, obj)
    return gi