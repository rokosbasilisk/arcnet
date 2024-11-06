from re_arc.dsl import *
def generate_1f876c06() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    succ = 0
    tr = 0
    maxtr = 10 * nlns
    direcs = ineighbors((0, 0))
    gi = canvas(bgc, (28, 11))
    inds = asindices(gi)
    while succ < nlns and tr < maxtr:
        tr += 1
        if len(inds) == 0:
            break
        lns = []
        for direc in direcs:
            ln = [loc]
            ofs = 1
            while True:
                nextpix = add(loc, multiply(ofs, direc))
                ofs += 1
                if nextpix not in inds:
                    break
                ln.append(nextpix)
            if len(ln) > 2:
                lns.append(ln)
        if len(lns) > 0:
            succ += 1
            lns = sorted(lns, key=len)
            ln = lns[idx]
            col = ccols[0]
            gi = fill(gi, col, {ln[0], ln[-1]})
            inds = inds - set(ln)
    return gi