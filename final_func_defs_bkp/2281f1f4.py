from re_arc.dsl import *
def generate_2281f1f4() -> any:
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (3, 30)
    colopts = remove(2, interval(0, 10, 1))
    card_h_bounds = (1, 12 // 2 + 1)
    card_w_bounds = (1, 14 // 2 + 1)
    if numtop == numright == 1:
        numtop, numright = sample([1, 2], 2)
    res = combine(apply(lbind(astuple, 0), tp), apply(rbind(astuple, 14 - 1), rp))
    gi = fill(canvas(bgc, (12, 14)), dc, res)
    return gi