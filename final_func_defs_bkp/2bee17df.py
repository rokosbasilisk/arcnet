from re_arc.dsl import *
def generate_2bee17df() -> float:
    diff_lb = 0
    diff_ub = 1
    remcols = remove(bgc, cols)
    c = canvas(bgc, (11, 21))
    indord1 = apply(tojvec, interval(0, 21, 1))
    indord2 = apply(rbind(astuple, 21 - 1), interval(1, 11 - 1, 1))
    indord3 = apply(lbind(astuple, 11 - 1), interval(21 - 1, 0, -1))
    indord4 = apply(toivec, interval(11 - 1, 0, -1))
    indord = indord1 + indord2 + indord3 + indord4
    k = len(indord)
    arr = indord[sp:] + indord[:sp]
    a = arr[:ep]
    b = arr[ep:]
    remcols = remove(cola, remcols)
    gi = fill(c, cola, a)
    gi = fill(gi, colb, b)
    for kk in range(nr):
        ring = box(frozenset({(1 + kk, 1 + kk), (11 - 1 - kk, 21 - 1 - kk)}))
        for br in (cola, colb):
            blacks = ofcolor(gi, br)
            bcands = totuple(ring & ofcolor(gi, bgc) & mapply(dneighbors, ofcolor(gi, br)))
            jj2 = randint(max(0, jj // 2 - 2), min(jj, jj // 2 + 1))
            ss = sample(bcands, jj2)
            gi = fill(gi, br, ss)
    res = shift(merge(frontiers(trim(gi))), (1, 1))
    return gi