from re_arc.dsl import *
def generate_469497ad() -> any:
    diff_lb = 0
    diff_ub = 1
    
    cols = remove(2, interval(0, 10, 1))
    
    
    bgc, sqc = sample(cols, 2)
    
    gi = canvas(bgc, (5, 3))
    
    
    
    
    
    sq = backdrop(frozenset({(sqloci, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)}))
    
    gi = fill(gi, sqc, sq)
    
    numcub = min(min(min(5, 3) + 1, 30 // (max(5, 3))), 7)
    
    numaccc = 4 - 1
    remcols = remove(bgc, remove(sqc, cols))
    
    gi = rot180(gi)
    
    
    for c, l in zip(ccols, locs):
        gi = fill(gi, c, shoot((0, l), (0, 1)))
        gi = fill(gi, c, shoot((l, 0), (1, 0)))
    
    gi = rot180(gi)
    
    return gi