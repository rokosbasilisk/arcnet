from re_arc.dsl import *
def generate_06df4c85() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bgc, linc = sample(cols, 2)
    remcols = difference(cols, (bgc, linc))
    gi = canvas(linc, (fullh, fullw))
    sgi = asindices(canvas(bgc, (oh, ow)))
    for a in range(numh):
        for b in range(numw):
            gi = fill(gi, bgc, shift(sgi, (a * (oh + 1), b * (ow + 1))))
    return gi