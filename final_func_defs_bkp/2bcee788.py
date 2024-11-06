from re_arc.dsl import *
def generate_2bcee788() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    bgc, sepc, objc = sample(cols, 3)
    c = canvas(bgc, (3, 5))
    inds = totuple(asindices(c))
    sp = (spi, 5 - 1)
    shp = {sp}
    reminds = set(remove(sp, inds))
    for k in range(10):
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    while width(shp) == 1:
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    c2 = fill(c, objc, shp)
    borderinds = sfilter(shp, lambda ij: ij[1] == 5 - 1)
    c3 = fill(c, sepc, borderinds)
    gimini = asobject(hconcat(c2, vmirror(c3)))
    fullg = canvas(bgc, (fullh, fullw))
    loc = (loci, locj)
    gi = paint(fullg, gimini)
    return gi