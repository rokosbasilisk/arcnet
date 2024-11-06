from re_arc.dsl import *
def generate_1c786137() -> float:
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (3, 30)
    num_cols_card_bounds = (1, 8)
    colopts = interval(1, 10, 1)
    noise_card_bounds = (0, 26 * 10)
    c = canvas(0, (26, 10))
    inds = totuple(asindices(c))
    noiseinds = sample(inds, num_noise)
    colset = sample(colopts, num_cols)
    trgcol = choice(difference(colopts, colset))
    noise = frozenset((choice(colset), ij) for ij in noiseinds)
    gi = paint(c, noise)
    boxhrng = (3, max(3, 26//2))
    boxwrng = (3, max(3, 10//2))
    loc = (boxi, boxj)
    llc = add(loc, toivec(boxh - 1))
    urc = add(loc, tojvec(boxw - 1))
    lrc = add(loc, (boxh - 1, boxw - 1))
    l1 = connect(loc, llc)
    l2 = connect(loc, urc)
    l3 = connect(urc, lrc)
    l4 = connect(llc, lrc)
    l = l1 | l2 | l3 | l4
    gi = fill(gi, trgcol, l)
    return gi