from re_arc.dsl import *
def generate_4938f0c2() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)
    remcols = remove(cc, remcols)
    sg = canvas(bgc, (oh, ow))
    locc = (oh - 1, ow - 1)
    sg = fill(sg, cc, {locc})
    reminds = totuple(remove(locc, asindices(sg)))
    cells = sample(reminds, ncells)
    while ncells == 4 and shape(cells) == (2, 2):
        cells = sample(reminds, ncells)
    sg = fill(sg, objc, cells)
    G1 = sg
    G2 = vmirror(sg)
    G3 = hmirror(sg)
    G4 = vmirror(hmirror(sg))
    vbar = canvas(bgc, (oh, 1))
    hbar = canvas(bgc, (1, ow))
    cp = canvas(cc, (1, 1))
    topg = hconcat(hconcat(G1, vbar), G2)
    botg = hconcat(hconcat(G3, vbar), G4)
    ggm = hconcat(hconcat(hbar, cp), hbar)
    GG = vconcat(vconcat(topg, ggm), botg)
    gg = asobject(GG)
    canv = canvas(bgc, (23, 22))
    loc = (loci, locj)
    gi = paint(canv, shift(asobject(sg), loc))
    gi = fill(gi, cc, ofcolor(gi, cc))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    ccpi, ccpj = center(ofcolor(gi, cc))
    gi = gi[:ccpi] + gi[ccpi+1:]
    gi = tuple(r[:ccpj] + r[ccpj + 1:] for r in gi)
    return gi