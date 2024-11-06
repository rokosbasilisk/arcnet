from re_arc.dsl import *
def generate_4612dd53() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    bgc, col = sample(cols, 2)
    onbx = totuple(crns)
    rembx = difference(bx, crns)
    onbr = sample(totuple(br), 2)
    rembr = difference(br, onbr)
    gi = fill(c, bgc, occbx)
    gi = fill(gi, bgc, occbr)
    if choice((True, False)):
        gi = fill(gi, bgc, br)
    return gi