from re_arc.dsl import *
def generate_0a938d79() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bgc, cola, colb = sample(cols, 3)
    gi = canvas(bgc, (6, 10))
    gi = fill(gi, cola, {(locia, locja)})
    gi = fill(gi, colb, {(locib, locjb)})
    ofs = -2 * (locja - locjb)
    rotf = choice((rot180, rot270))
    gi = rotf(gi)
    return gi