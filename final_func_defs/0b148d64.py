from re_arc.dsl import *
def generate_0b148d64() -> float:
    diff_lb = 0
    diff_ub = 1
    itv = interval(0, 10, 1)
    remitv = remove(bgc, itv)
    A = backdrop(frozenset({(0, 0), (x, y)}))
    B = backdrop(frozenset({(x + di, 0), (7 - 1, y)}))
    C = backdrop(frozenset({(0, y + dj), (x, 26 - 1)}))
    D = backdrop(frozenset({(x + di, y + dj), (7 - 1, 26 - 1)}))
    rem = remove(trg, (A, B, C, D))
    subf = lambda bx: {
        choice(totuple(connect(ulcorner(bx), urcorner(bx)))),
        choice(totuple(connect(ulcorner(bx), llcorner(bx)))),
        choice(totuple(connect(urcorner(bx), lrcorner(bx)))),
        choice(totuple(connect(llcorner(bx), lrcorner(bx)))),
    }
    sampler = lambda bx: set(sample(
        totuple(bx),
        len(bx) - unifint(diff_lb, diff_ub, (0, len(bx) - 1))
    ))
    gi = fill(canvas(bgc, (7, 26)), cola, sampler(trg) | subf(trg))
    for r in rem:
        gi = fill(gi, colb, sampler(r) | subf(r))
    return gi