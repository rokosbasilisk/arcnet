from re_arc.dsl import *
def generate_27a28665() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    mapping = [
        (1, {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}),
        (2, {(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)}),
        (3, {(2, 0), (0, 1), (0, 2), (1, 1), (1, 2)}),
        (6, {(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)})
    ]
    col, obj = choice(mapping)
    bgc, objc = sample(cols, 2)
    gi = canvas(bgc, (25, 29))
    canv = canvas(bgc, (3, 3))
    canv = fill(canv, objc, obj)
    canv = upscale(canv, fac)
    obj = asobject(canv)
    loc = (loci, locj)
    gi = paint(gi, shift(obj, loc))
    return gi