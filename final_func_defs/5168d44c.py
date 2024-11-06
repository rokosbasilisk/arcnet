from re_arc.dsl import *
def generate_5168d44c() -> any:
    diff_lb = 0
    diff_ub = 1
    
    cols = interval(0, 10, 1)
    
    
    
    
    
    
    
    
    
    dotloc = (dotloci, dotlocj)
    
    remcols = remove(bgc, cols)
    
    remcols = remove(dotcol, remcols)
    
    gi = canvas(bgc, (21, 20))
    dotshap = (doth, dotw)
    starterdot = backdrop(frozenset({dotloc, add(dotloc, decrement(dotshap))}))
    bordershap = (borderh, borderw)
    offset = add(multiply(direc, dotshap), multiply(direc, bordershap))
    itv = interval(-15, 16, 1)
    itv = apply(lbind(multiply, offset), itv)
    dots = mapply(lbind(shift, starterdot), itv)
    gi = fill(gi, dotcol, dots)
    
    protobx = backdrop(frozenset({
        (dotloci - borderh, dotlocj - borderw),
        (dotloci + doth + borderh - 1, dotlocj + dotw + borderw - 1),
    }))
    
    bx = protobx - starterdot
    bxshifted = shift(bx, offset)
    gi = fill(gi, boxcol, bx)
    
    return gi