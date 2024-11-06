from re_arc.dsl import *
def generate_363442ee() -> any:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    rss = (rsh, rsw)
    remcols = remove(bgc, cols)
    remcols = remove(barcol, remcols)
    rsi = canvas(bgc, rss)
    rso = canvas(bgc, rss)
    ls = canvas(bgc, ((nremh - 1) * 1, 3))
    ulc = canvas(bgc, (1, 3))
    bar = canvas(barcol, (nremh * 1, 1))
    dotcands = totuple(product(interval(0, rsh, 1), interval(0, rsw, 3)))
    ndots = choice((dev, len(dotcands) - dev))
    ndots = min(max(1, ndots), len(dotcands))
    dots = sample(dotcands, ndots)
    fullremcols = sample(remcols, nfullremcols)
    for ij in asindices(ulc):
        ulc = fill(ulc, choice(fullremcols), {ij})
    ulco = asobject(ulc)
    osf = (1//2, 3//2)
    for d in dots:
        rsi = fill(rsi, dotcol, {add(osf, d)})
        rso = paint(rso, shift(ulco, d))
    gi = hconcat(hconcat(vconcat(ulc, ls), bar), rsi)
    return gi