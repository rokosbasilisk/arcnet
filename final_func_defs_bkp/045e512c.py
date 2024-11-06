from re_arc.dsl import *
def generate_045e512c() -> None:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    while True:
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {c1, c2, c3, c4}
        remcands = totuple(bounds - obj)
        for k in range(ncells):
            obj.add(loc)
            remcands = remove(loc, remcands)
        objt = normalize(obj)
        cc = canvas(0, shape(obj))
        cc = fill(cc, 1, objt)
        if len(colorfilter(objects(cc, T, T, F), 1)) == 1:
            break
    bgc, objc = sample(cols, 2)
    remcols = remove(bgc, remove(objc, cols))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (17, 25))
    obj = shift(recolor(objc, obj), loc)
    gi = paint(gi, obj)
    options = totuple(neighbors((0, 0)))
    dirs = sample(options, ndirs)
    dcols = [choice(ccols) for k in range(ndirs)]
    hbars = hfrontier((loci - 2, 0)) | hfrontier((loci + oh + 1, 0))
    vbars = vfrontier((0, locj - 2)) | vfrontier((0, locj + ow + 1))
    bars = hbars | vbars
    ofs = increment((oh, ow))
    for direc, col in zip(dirs, dcols):
        indicatorobj = shift(obj, multiply(direc, increment((oh, ow))))
        indicatorobj = sfilter(indicatorobj, lambda cij: cij[1] in bars)
        nindsd = unifint(diff_lb, diff_ub, (0, len(indicatorobj) - 1))
        ninds = len(indicatorobj) - nindsd
        indicatorobj = set(sample(totuple(indicatorobj), ninds))
        if len(indicatorobj) > 0 and len(indicatorobj) < len(obj):
            gi = fill(gi, col, indicatorobj)
    return gi