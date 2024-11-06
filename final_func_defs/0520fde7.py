from re_arc.dsl import *
def generate_0520fde7() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    bgc = 0
    remcols = remove(bgc, cols)
    remcols = remove(barcol, remcols)
    canv = canvas(bgc, (19, 11))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (19, 1))
    mp = (19 * 11) // 2
    devrng = (0, mp)
    numa = mp + deva
    numb = mp + devb
    numa = max(min(19 * 11 - 1, numa), 1)
    numb = max(min(19 * 11 - 1, numb), 1)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    return gi