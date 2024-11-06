from re_arc.dsl import *
def generate_5117e062() -> float:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    tr = 0
    maxtr = 4 * nobjs
    done = False
    succ = 0
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (21, 16))
    inds = asindices(gi)
    while tr < maxtr and succ < nobjs:
        bx = asindices(canvas(-1, (oh, ow)))
        bx = remove(sp, bx)
        obj = {sp}
        for k in range(nc - 1):
            obj.add(choice(totuple((bx - obj) & mapply(neighbors, obj))))
        if not done:
            done = True
            obj2 = {(coll, idx)}
            obj3 = recolor(coll2, remove(idx, obj))
            obj = obj2 | obj3
        else:
            obj = recolor(choice(remcols), obj)
        locopts = sfilter(inds, lambda ij: ij[0] <= 21 - oh and ij[1] <= 16 - ow)
        tr += 1
        if len(locopts) == 0:
            continue
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            gi = paint(gi, plcd)
            succ += 1
            inds = (inds - plcdi) - mapply(neighbors, plcdi)
    return gi