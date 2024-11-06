from re_arc.dsl import *
def generate_3aa6fb7a() -> float:
    diff_lb = 0
    diff_ub = 1
    base = (ORIGIN, RIGHT, DOWN, UNITY)
    cols = remove(1, interval(0, 10, 1))
    remcols = remove(bgc, cols)
    inds = totuple(asindices(gi))
    maxnum = ((25 * 28) // 2) // 3
    kk, tr = 0, 0
    maxtrials = num * 2
    binds = set()
    while kk < num and tr < maxtrials:
        oo = remove(ooo, base)
        if set(oop).issubset(inds):
            inds = difference(inds, totuple(combine(oop, totuple(mapply(dneighbors, oop)))))
            binds.add(add(ooo, loc))
            kk += 1
        tr += 1
    return gi