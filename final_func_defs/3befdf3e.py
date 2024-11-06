from re_arc.dsl import *
def generate_3befdf3e() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    succ = 0
    maxtr = 5 * nobjs
    tr = 0
    gi = canvas(bgc, (19, 16))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        if len(inds) == 0:
            break
        cands = sfilter(inds, lambda ij: ij[0] <= 19 - fullh and ij[1] <= 16 - fullw)
        if len(cands) == 0:
            continue
        loci, locj = loc
        fullobj = backdrop(frozenset({loc, (loci + fullh - 1, locj + fullw - 1)}))
        if fullobj.issubset(inds):
            succ += 1
            inds = inds - fullobj
            incol, outcol = sample(ccols, 2)
            ofincol = backdrop(frozenset({(loci + rh + 1, locj + rw + 1), (loci + 2 * rh, locj + 2 * rw)}))
            ofoutcol = outbox(ofincol)
            gi = fill(gi, incol, ofincol)
            gi = fill(gi, outcol, ofoutcol)
            ilocs = apply(first, ofoutcol)
            jlocs = apply(last, ofoutcol)
            ff = lambda ij: ij[0] in ilocs or ij[1] in jlocs
            addon = sfilter(fullobj - (ofincol | ofoutcol), ff)
            gi = fill(gi, outcol, addon)
    return gi