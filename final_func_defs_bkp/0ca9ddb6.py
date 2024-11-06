from re_arc.dsl import *
def generate_0ca9ddb6() -> any:
    cols = difference(interval(0, 10, 1), (1, 2, 4, 6, 7, 8))
    xi = {(8, (0, 0))}
    xo = {(8, (0, 0))}
    ai = {(6, (0, 0))}
    ao = {(6, (0, 0))}
    bi = {(2, (1, 1))}
    bo = {(2, (1, 1))} | recolor(4, ineighbors((1, 1)))
    ci = {(1, (1, 1))}
    co = {(1, (1, 1))} | recolor(7, dneighbors((1, 1)))
    arr = ((ai, ao), (bi, bo), (ci, co), (xi, xo))    
    maxtr = 5 * nobjs
    tr = 0
    succ = 0
    bgc = choice(cols)
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        ino, outo = choice(arr)
        loc = choice(totuple(inds))
        oplcd = shift(outo, loc)
        oplcdi = toindices(oplcd)
        if oplcdi.issubset(inds):
            succ += 1
            inds = inds - oplcdi
        tr += 1
    return gi