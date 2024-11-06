from re_arc.dsl import *
def generate_4be741c5() -> list:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    gi = merge(tuple(repeat(repeat(c, 20), 3) for c in ccols))
    while len(gi) < 30:
        gi = gi[:idx] + gi[idx:idx+1] + gi[idx:]
    gi = dmirror(gi)
    for k in range(ndisturbances):
        options = []
        for a in range(20):
            for b in range(30 - 3):
                if gi[a][b] == gi[a][b+1] and gi[a][b+2] == gi[a][b+3]:
                    options.append((a, b, gi[a][b], gi[a][b+2]))
        if len(options) == 0:
            break
        a, b, c1, c2 = choice(options)
        if choice((True, False)):
            gi = fill(gi, c2, {(a, b+1)})
        else:
            gi = fill(gi, c1, {(a, b+2)})
    if choice((True, False)):
        gi = dmirror(gi)
    return gi