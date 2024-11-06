from re_arc.dsl import *
def generate_11852cab() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    r1 = ((0, 0), (0, 4), (4, 0), (4, 4))
    r2 = ((2, 0), (0, 2), (4, 2), (2, 4))
    r3 = ((1, 1), (3, 1), (1, 3), (3, 3))
    r4 = ((2, 2),)
    rings = [r4, r3, r2, r1]
    bx = backdrop(frozenset(r1))
    remcols = remove(bgc, cols)
    ccols = sample(remcols, 2)
    gi = canvas(bgc, (16, 29))
    inds = shift(asindices(trim(gi)), UNITY)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (16 * 29) // 36)))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= 16 - 5 and ij[0] <= 29 - 5)
        if len(cands) == 0:
            break
        plcd = shift(bx, loc)
        if plcd.issubset(inds):
            inds = (inds - plcd) - outbox(plcd)
            ringcols = [choice(ccols) for k in range(4)]
            plcdrings = [shift(r, loc) for r in rings]
            gi = fill(gi, ringcols[0], plcdrings[0])
            gi = fill(gi, ringcols[idx], plcdrings[idx])
            remrings = plcdrings[1:idx] + plcdrings[idx+1:]
            remringcols = ringcols[1:idx] + ringcols[idx+1:]
            locs = sample((0, 1), numrs)
            remrings = [rr for j, rr in enumerate(remrings) if j in locs]
            remringcols = [rr for j, rr in enumerate(remringcols) if j in locs]
            tofillgi = merge(frozenset(
                recolor(col, frozenset(sample(totuple(remring), 4 - unifint(diff_lb, diff_ub, (0, 3))))) for remring, col in zip(remrings, remringcols)
            ))
            if min(shape(tofillgi)) == 5:
                succ += 1
                gi = paint(gi, tofillgi)
    return gi