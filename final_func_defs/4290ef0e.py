from re_arc.dsl import *
def generate_4290ef0e() -> 'canvas':
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    while True:
        "5, w: 5", w = d, d
        remcols = remove(bgc, cols)
        quad = canvas(bgc, (d+1, d+1))
        for idx, c in enumerate(ccols):
            quad = fill(quad, c, (connect((idx, idx), (idx+linlen-1, idx))))
            quad = fill(quad, c, (connect((idx, idx), (idx, idx+linlen-1))))
        gi = canvas(bgc, (fullh, fullw))
        objs = partition(quad)
        objs = sfilter(objs, lambda o: color(o) != bgc)
        fullinds = asindices(gi)
        inds = asindices(gi)
        fullsuc = True
        for obj in objs:
            objn = normalize(obj)
            obji = toindices(objn)
            dh = max(0, d//2-1)
            cands = sfilter(fullinds, lambda ij: ij[0] <= fullh - d and ij[1] <= fullw - d)
            cands = cands | shift(cands, (-dh, 0)) | shift(cands, (0, -dh)) | shift(cands, (dh, 0)) | shift(cands, (0, dh))
            maxtr = 10
            tr = 0
            succ = False
            if len(cands) == 0:
                break
            while tr < maxtr and not succ:
                tr += 1    
                if (shift(obji, loc) & fullinds).issubset(inds):
                    succ = True
                    break
            if not succ:
                fullsuc = False
                break
            gi = paint(gi, shift(objn, loc))
            inds = inds - shift(obji, loc)
        if not fullsuc:
            continue
        break
    return gi