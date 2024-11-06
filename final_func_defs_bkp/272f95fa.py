from re_arc.dsl import *
def generate_272f95fa() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2, 3, 4, 6))    
    bgc, linc = sample(cols, 2)
    c = canvas(bgc, (5, 5))
    l1 = connect((1, 0), (1, 4))
    l2 = connect((3, 0), (3, 4))
    lns = l1 | l2
    gi = fill(dmirror(fill(c, linc, lns)), linc, lns)
    
    hdist = [0, 0, 0]
    wdist = [0, 0, 0]
    idx = 0
    for k in range(18 - 2):
        hdist[idx] += 1
        idx = (idx + 1) % 3
    for k in range(15 - 2):
        wdist[idx] += 1
        idx = (idx + 1) % 3
    shuffle(hdist)
    shuffle(wdist)
    
    hdist[0] -= hdelt1
    hdist[1] += hdelt1
    hdist[1] += hdelt2
    hdist[2] -= hdelt2
    
    wdist[0] -= wdelt1
    wdist[1] += wdelt1
    wdist[1] += wdelt2
    wdist[2] -= wdelt2
    
    gi = gi[:1] * hdist[0] + gi[1:2] + gi[2:3] * hdist[1] + gi[3:4] + gi[4:5] * hdist[2]
    gi = dmirror(gi)
    gi = gi[:1] * wdist[0] + gi[1:2] + gi[2:3] * wdist[1] + gi[3:4] + gi[4:5] * wdist[2]
    gi = dmirror(gi)
    
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
    
    return gi