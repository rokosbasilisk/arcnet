from re_arc.dsl import *
def generate_4522001f() -> any:
    diff_lb = 0
    diff_ub = 1
    
    cols = interval(0, 10, 1)
    
    
    bgc, sqc, dotc = sample(cols, 3)
    
    
    sqi = {(dotc, (1, 1))} | recolor(sqc, {(0, 0), (0, 1), (1, 0)})
    sqo = backdrop(frozenset({(0, 0), (3, 3)}))
    sqo |= shift(sqo, (4, 4))
    
    
    
    loc = (loci, locj)
    plcdi = shift(sqi, loc)
    plcdo = shift(sqo, loc)
    
    
    
    succ = 0
    tr = 0
    maxtr = 10 * noccs
    iinds = ofcolor(gi, bgc) - mapply(dneighbors, toindices(plcdi))
    
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = sfilter(iinds, lambda ij: ij[0] <= 5 - 2 and ij[1] <= 9 - 2)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcdi = shift(sqi, loc)
        plcdo = shift(sqo, loc)
        plcdii = toindices(plcdi)
        if plcdii.issubset(iinds):
            succ += 1
            iinds = (iinds - plcdii) - mapply(dneighbors, plcdii)
    
    rotf = choice((identity, rot90, rot180, rot270))
    
    return gi