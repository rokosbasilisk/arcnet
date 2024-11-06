from re_arc.dsl import *
def generate_2204b7a8() -> any:
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (4, 30)
    colopts = interval(0, 10, 1)
    
    while True:
        
        
        
        remcols = remove(bgc, colopts)
        c = canvas(bgc, (14, 17))
        inds = totuple(shift(asindices(canvas(0, (14, 17 - 2))), RIGHT))
        
        
        remcols2 = remove(ccol, remcols)
        
        
        nc_bounds = (1, (14 * (17 - 2)) // 2 - 1)
        
        
        if 17 % 2 == 1:
        
        gi = fill(c, c1, vfrontier(ORIGIN))
        gi = fill(gi, c2, vfrontier(tojvec(17 - 1)))
        gi = fill(gi, ccol, locs)
        
        if len(palette(gi)) == 4:
            break
    
    if choice((True, False)):
        gi = dmirror(gi)
    
    return gi