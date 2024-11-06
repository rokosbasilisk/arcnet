from re_arc.dsl import *
def generate_3f7978a0() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bgc, noisec, linec = sample(cols, 3)
    c = canvas(bgc, (12, 19))
    inds = totuple(asindices(c))
    noise = sample(inds, nnoise)
    gi = fill(c, noisec, noise)
    ulc = (loci, locj)
    lrc = (loci + oh - 1, locj + ow - 1)
    llc = (loci + oh - 1, locj)
    urc = (loci, locj + ow - 1)
    gi = fill(gi, linec, connect(ulc, llc))
    gi = fill(gi, linec, connect(urc, lrc))
    crns = {ulc, lrc, llc, urc}
    gi = fill(gi, noisec, crns)
    return gi