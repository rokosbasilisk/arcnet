from re_arc.dsl import *
def generate_23b5c85d() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    colopts = remove(bgc, cols)
    gi = canvas(bgc, (9, 20))
    cnt = 0
    while cnt < num:
        colopts = remove(col, colopts)
        obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        gi2 = fill(gi, col, obj)
        if color(argmin(sfilter(partition(gi2), fork(equality, size, fork(multiply, height, width))), fork(multiply, height, width))) != col:
            break
        else:
            gi = gi2
        if oh < 1 or ow < 1:
            break
        cnt += 1
    return gi