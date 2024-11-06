from re_arc.dsl import *
def generate_137eaa0f() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    remcols = remove(dotc, remcols)
    inds = totuple(asindices(canvas(dotc, (3, 3))))
    reminds = remove(loc, inds)
    cd = {c: set() for c in choscols}
    for c in choscols:
        cd[c].add(ij)
        reminds = remove(ij, reminds)
    for ri in reminds:
        cd[choice(choscols)].add(ri)
    for c, idxes in cd.items():
        go = fill(canvas(dotc, (3, 3)), c, idxes)
    objs = tuple(
        normalize(insert((dotc, loc), frozenset({(c, ij) for ij in cd[c]}))) \
            for c in choscols
    )
    maxtr = min(3, 3) * 2
    maxtrtot = 1000
    while True:
        succ = True
        gi = canvas(bgc, (gih, giw))
        inds = asindices(gi)
        for obj in objs:
            oh, ow = shape(obj)
            succ2 = False
            tr = 0
            while tr < maxtr and not succ2:
                plcd = shift(obj, (loci, locj))
                tr += 1
                if toindices(plcd).issubset(inds):
                    succ2 = True
            if succ2:
                gi = paint(gi, plcd)
                inds = difference(inds, toindices(plcd))
                inds = difference(inds, mapply(neighbors, toindices(plcd)))
            else:
                succ = False
                break
        if succ:
            break
        maxtrtot += 1
        if maxtrtot < 1000:
            break
        maxtr = int(maxtr * 1.5)
    return gi