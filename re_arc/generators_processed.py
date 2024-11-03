from dsl import *


def generate_dbc1a6ce():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (3, 30)
    colopts = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgc = choice(colopts)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    card_bounds = (0, max(1, (h * w) // 4))
    num = unifint(diff_lb, diff_ub, card_bounds)
    s = sample(inds, num)
    fgcol = choice(remove(bgc, colopts))
    gi = fill(c, fgcol, s)
    return gi


def generate_2281f1f4():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (3, 30)
    colopts = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    card_h_bounds = (1, h // 2 + 1)
    card_w_bounds = (1, w // 2 + 1)
    numtop = unifint(diff_lb, diff_ub, card_w_bounds)
    numright = unifint(diff_lb, diff_ub, card_h_bounds)
    if numtop == numright == 1:
        numtop, numright = sample([1, 2], 2)
    tp = sample(interval(0, w - 1, 1), numtop)
    rp = sample(interval(1, h, 1), numright)
    res = combine(apply(lbind(astuple, 0), tp), apply(rbind(astuple, w - 1), rp))
    bgc = choice(colopts)
    dc = choice(remove(bgc, colopts))
    gi = fill(canvas(bgc, (h, w)), dc, res)
    return gi


def generate_c1d99e64():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (4, 30)
    colopts = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    nofrontcol = choice(colopts)
    noisefrontcol = choice(remove(nofrontcol, colopts))
    gi = canvas(nofrontcol, (h, w))
    return gi


def generate_623ea044():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (3, 30)
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgc = choice(colopts)
    g = canvas(bgc, (h, w))
    fullinds = asindices(g)
    inds = totuple(asindices(g))
    card_bounds = (0, max(int(h * w * 0.1), 1))
    numdots = unifint(diff_lb, diff_ub, card_bounds)
    dots = sample(inds, numdots)
    gi = canvas(bgc, (h, w))
    return gi


def generate_1190e5a7():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (3, 30)
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgc = choice(colopts)
    c = canvas(bgc, (h, w))
    nhf_bounds = (1, h // 3)
    nvf_bounds = (1, w // 3)
    nhf = unifint(diff_lb, diff_ub, nhf_bounds)
    nvf = unifint(diff_lb, diff_ub, nvf_bounds)
    hf_options = interval(1, h - 1, 1)
    vf_options = interval(1, w - 1, 1)
    hf_selection = []
    for k in range(nhf):
        hf = choice(hf_options)
        hf_selection.append(hf)
        hf_options = difference(hf_options, (hf - 1, hf, hf + 1))
    vf_selection = []
    for k in range(nvf):
        vf = choice(vf_options)
        vf_selection.append(vf)
        vf_options = difference(vf_options, (vf - 1, vf, vf + 1))
    remcols = remove(bgc, colopts)
    rcf = lambda x: recolor(choice(remcols), x)
    hfs = mapply(chain(rcf, hfrontier, toivec), tuple(hf_selection))
    vfs = mapply(chain(rcf, vfrontier, tojvec), tuple(vf_selection))
    gi = paint(c, combine(hfs, vfs))
    return gi


def generate_5614dbcf():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (2, 10)
    col_card_bounds = (1, 8)
    noise_card_bounds = (0, 8)
    colopts = remove(5, interval(1, 10, 1))
    noisedindscands = totuple(asindices(canvas(0, (3, 3))))
    d = unifint(diff_lb, diff_ub, dim_bounds)
    cells_card_bounds = (1, d * d)
    go = canvas(0, (d, d))
    inds = totuple(asindices(go))
    numocc = unifint(diff_lb, diff_ub, cells_card_bounds)
    numcol = unifint(diff_lb, diff_ub, col_card_bounds)
    occs = sample(inds, numocc)
    colset = sample(colopts, numcol)
    gi = upscale(go, THREE)
    return gi


def generate_05269061():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (2, 30)
    colopts = interval(1, 10, 1)
    d = unifint(diff_lb, diff_ub, dim_bounds)
    go = canvas(0, (d, d))
    gi = canvas(0, (d, d))
    return gi


def generate_1c786137():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (3, 30)
    num_cols_card_bounds = (1, 8)
    colopts = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    noise_card_bounds = (0, h * w)
    c = canvas(0, (h, w))
    inds = totuple(asindices(c))
    num_noise = unifint(diff_lb, diff_ub, noise_card_bounds)
    num_cols = unifint(diff_lb, diff_ub, num_cols_card_bounds)
    noiseinds = sample(inds, num_noise)
    colset = sample(colopts, num_cols)
    trgcol = choice(difference(colopts, colset))
    noise = frozenset((choice(colset), ij) for ij in noiseinds)
    gi = paint(c, noise)
    return gi


def generate_2204b7a8():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (4, 30)
    colopts = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, dim_bounds)
        w = unifint(diff_lb, diff_ub, dim_bounds)
        bgc = choice(colopts)
        remcols = remove(bgc, colopts)
        c = canvas(bgc, (h, w))
        inds = totuple(shift(asindices(canvas(0, (h, w - 2))), RIGHT))
        ccol = choice(remcols)
        remcols2 = remove(ccol, remcols)
        c1 = choice(remcols2)
        c2 = choice(remove(c1, remcols2))
        nc_bounds = (1, (h * (w - 2)) // 2 - 1)
        nc = unifint(diff_lb, diff_ub, nc_bounds)
        locs = sample(inds, nc)
        if w % 2 == 1:
            locs = difference(locs, vfrontier(tojvec(w // 2)))
        gi = fill(c, c1, vfrontier(ORIGIN))
        return gi


def generate_23581191():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (3, 30)
    colopts = remove(2, interval(0, 10, 1))
    f = fork(combine, hfrontier, vfrontier)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgcol = choice(colopts)
    remcols = remove(bgcol, colopts)
    c = canvas(bgcol, (h, w))
    inds = totuple(asindices(c))
    acol = choice(remcols)
    bcol = choice(remove(acol, remcols))
    card_bounds = (1, (h * w) // 4)
    na = unifint(diff_lb, diff_ub, card_bounds)
    nb = unifint(diff_lb, diff_ub, card_bounds)
    a = sample(inds, na)
    b = sample(difference(inds, a), nb)
    gi = fill(c, acol, a)
    return gi


def generate_8be77c9e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_6d0aefbc():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_74dd1130():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_62c24649():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_6150a2bd():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_6fa7a44f():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_8d5021e8():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 10))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_0520fde7():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = 0
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    cola = choice(remcols)
    colb = choice(remcols)
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    return gi


def generate_46442a0e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = h
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_1b2d62fb():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = 0
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    cola = choice(remcols)
    colb = choice(remcols)
    canv = canvas(0, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    return gi


def generate_3428a4f5():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 14))
    bgc = 0
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    cola = choice(remcols)
    colb = choice(remcols)
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    return gi


def generate_42a50994():
    diff_lb = 0
    diff_ub = 1
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(colopts)
    remcols = remove(bgc, colopts)
    c = canvas(bgc, (h, w))
    card_bounds = (0, max(0, (h * w) // 2 - 1))
    num = unifint(diff_lb, diff_ub, card_bounds)
    numcols = unifint(diff_lb, diff_ub, (0, min(9, num)))
    inds = totuple(asindices(c))
    chosinds = sample(inds, num)
    choscols = sample(remcols, numcols)
    locs = interval(0, len(chosinds), 1)
    choslocs = sample(locs, numcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_08ed6ac7():
    diff_lb = 0
    diff_ub = 1
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc = choice(difference(colopts, (1, 2, 3, 4)))
    remcols = remove(bgc, colopts)
    gi = canvas(bgc, (h, w))
    return gi


def generate_8f2ea7aa():
    diff_lb = 0
    diff_ub = 1
    colopts = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 5))
    bgc = choice(colopts)
    remcols = remove(bgc, colopts)
    d2 = d ** 2
    gi = canvas(bgc, (d2, d2))
    return gi


def generate_7fe24cdd():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = h
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_85c4e7cd():
    diff_lb = 0
    diff_ub = 1
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 15))
    ncols = unifint(diff_lb, diff_ub, (1, 10))
    cols = sample(colopts, ncols)
    colord = [choice(cols) for j in range(min(h, w))]
    shp = (h*2, w*2)
    gi = canvas(0, shp)
    return gi


def generate_8e5a5113():
    diff_lb = 0
    diff_ub = 1
    colopts = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 9))
    bgc = choice(colopts)
    remcols = remove(bgc, colopts)
    k = 4 if d < 7 else 3
    nbound = (2, k)
    num = unifint(diff_lb, diff_ub, nbound)
    rotfs = (identity, rot90, rot180, rot270)
    barc = choice(remcols)
    remcols = remove(barc, remcols)
    colbnds = (1, 8)
    ncols = unifint(diff_lb, diff_ub, colbnds)
    patcols = sample(remcols, ncols)
    bgcanv = canvas(bgc, (d, d))
    c = canvas(bgc, (d, d))
    inds = totuple(asindices(c))
    ncolbnds = (1, d ** 2 - 1)
    ncells = unifint(diff_lb, diff_ub, ncolbnds)
    indsss = sample(inds, ncells)
    for ij in indsss:
        c = fill(c, choice(patcols), {ij})
    barr = canvas(barc, (d, 1))
    fillinidx = choice(interval(0, num, 1))
    gi = rot90(rot270(c if fillinidx == 0 else bgcanv))
    return gi


def generate_4c4377d9():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_a65b410d():
    diff_lb = 0
    diff_ub = 1
    colopts = difference(interval(0, 10, 1), (1, 3))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    mpi = h // 2
    mpj = w // 2
    devi = unifint(diff_lb, diff_ub, (0, mpi))
    devj = unifint(diff_lb, diff_ub, (0, mpj))
    if choice((True, False)):
        locj = devj
        loci = devi
    else:
        loci = h - devi
        locj = w - devj
    loci = max(min(h - 2, loci), 1)
    locj = max(min(w - 2, locj), 1)
    loc = (loci, locj)
    bgc = choice(colopts)
    linc = choice(remove(bgc, colopts))
    gi = canvas(bgc, (h, w))
    return gi


def generate_5168d44c():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    doth = unifint(diff_lb, diff_ub, (1, h//3))
    dotw = unifint(diff_lb, diff_ub, (1, w//3))
    borderh = unifint(diff_lb, diff_ub, (1, h//4))
    borderw = unifint(diff_lb, diff_ub, (1, w//4))
    direc = choice((DOWN, RIGHT, UNITY))
    dotloci = randint(0, h - doth - 1 if direc == RIGHT else h - doth - borderh - 1)
    dotlocj = randint(0, w - dotw - 1 if direc == DOWN else w - dotw - borderw - 1)
    dotloc = (dotloci, dotlocj)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    dotcol = choice(remcols)
    remcols = remove(dotcol, remcols)
    boxcol = choice(remcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_a9f96cdd():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (3, 6, 7, 8))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    fgc = choice(remove(bgc, cols))
    gi = canvas(bgc, (h, w))
    return gi


def generate_9172f3a0():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 10))
    w = unifint(diff_lb, diff_ub, (1, 10))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_67a423a3():
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_db3e9e38():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    barth = unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
    loci = unifint(diff_lb, diff_ub, (1, h - 2))
    locj = randint(1, w - barth - 1)
    bar = backdrop(frozenset({(loci, locj), (0, locj + barth - 1)}))
    gi = canvas(bgc, (h, w))
    return gi


def generate_9dfd6313():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    dh = unifint(diff_lb, diff_ub, (1, 14))
    d = 2 * dh + 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    gi = canvas(bgc, (d, d))
    return gi


def generate_746b3537():
    diff_lb = 0
    diff_ub = 1
    fullcols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (1, 30))
    cols = []
    lastc = -1
    for k in range(h):
        c = choice(remove(lastc, fullcols))
        cols.append(c)
        lastc = c
    go = tuple((c,) for c in cols)
    gi = tuple(repeat(c, w) for c in cols)
    return gi


def generate_75b8110e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (2, 15))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c1, c2, c3, c4 = sample(remcols, 4)
    canv = canvas(bgc, (h, w))
    cels = totuple(asindices(canv))
    mp = (h * w) // 2
    nums = []
    for k in range(4):
        dev = unifint(diff_lb, diff_ub, (0, mp))
        if choice((True, False)):
            num = h * w - dev
        else:
            num = dev
        num = min(max(0, num), h * w - 1)
        nums.append(num)
    s1, s2, s3, s4 = [sample(cels, num) for num in nums]
    gi1 = fill(canv, c1, s1)
    gi2 = fill(canv, c2, s2)
    gi3 = fill(canv, c3, s3)
    gi4 = fill(canv, c4, s4)
    gi = vconcat(hconcat(gi1, gi2), hconcat(gi3, gi4))
    return gi


def generate_1cf80156():
    diff_lb = 0
    diff_ub = 1
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(colopts)
    fgc = choice(remove(bgc, colopts))
    gi = canvas(bgc, (h, w))
    return gi


def generate_28bf18c6():
    diff_lb = 0
    diff_ub = 1
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(colopts)
    fgc = choice(remove(bgc, colopts))
    gi = canvas(bgc, (h, w))
    return gi


def generate_22eb0ac0():
    diff_lb = 0
    diff_ub = 1
    colopts = interval(0, 10, 1)
    gi = canvas(0, (1, 1))
    return gi


def generate_4258a5f9():
    diff_lb = 0
    diff_ub = 1
    colopts = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc = choice(colopts)
    remcols = remove(bgc, colopts)
    fgc = choice(remcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_1e0a9b12():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    ff = chain(dmirror, lbind(apply, rbind(order, identity)), dmirror)
    while True:
        h = unifint(diff_lb, diff_ub, (3, 30))
        w = unifint(diff_lb, diff_ub, (3, 30))
        nc = unifint(diff_lb, diff_ub, (1, w))
        bgc = choice(cols)
        gi = canvas(bgc, (h, w))
        return gi


def generate_9565186b():
    diff_lb = 0
    diff_ub = 1
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    wg = canvas(5, (h, w))
    numcols = unifint(diff_lb, diff_ub, (2, min(h * w - 1, 8)))
    mostcol = choice(cols)
    nummostcol_lb = (h * w) // numcols + 1
    nummostcol_ub = h * w - numcols + 1
    ubmlb = nummostcol_ub - nummostcol_lb
    nmcdev = unifint(diff_lb, diff_ub, (0, ubmlb))
    nummostcol = nummostcol_ub - nmcdev
    nummostcol = min(max(nummostcol, nummostcol_lb), nummostcol_ub)
    inds = totuple(asindices(wg))
    mostcollocs = sample(inds, nummostcol)
    gi = fill(wg, mostcol, mostcollocs)
    return gi


def generate_6e02f1e3():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (3, 30))
    c = canvas(0, (d, d))
    inds = list(asindices(c))
    shuffle(inds)
    num = d ** 2
    numcols = choice((1, 2, 3))
    chcols = sample(cols, numcols)
    if len(chcols) == 1:
        gi = canvas(chcols[0], (d, d))
        return gi


def generate_2dc579da():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    dotc = choice(remcols)
    hdev = unifint(diff_lb, diff_ub, (0, (h - 2) // 2))
    lineh = choice((hdev, h - 2 - hdev))
    lineh = max(min(h - 2, lineh), 1)
    wdev = unifint(diff_lb, diff_ub, (0, (w - 2) // 2))
    linew = choice((wdev, w - 2 - wdev))
    linew = max(min(w - 2, linew), 1)
    locidev = unifint(diff_lb, diff_ub, (1, h // 2))
    loci = choice((h // 2 - locidev, h // 2 + locidev))
    loci = min(max(1, loci), h - lineh - 1)
    locjdev = unifint(diff_lb, diff_ub, (1, w // 2))
    locj = choice((w // 2 - locjdev, w // 2 + locjdev))
    locj = min(max(1, locj), w - linew - 1)
    gi = canvas(bgc, (h, w))
    return gi


def generate_2dee498d():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 10))
    bgc = choice(cols)
    go = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(go))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        go = fill(go, col, chos)
        inds = difference(inds, chos)
    gi = hconcat(go, hconcat(go, go))
    return gi


def generate_508bd3b6():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (h, 30))
    barh = unifint(diff_lb, diff_ub, (1, h // 2))
    barloci = unifint(diff_lb, diff_ub, (2, h - barh))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barc = choice(remcols)
    remcols = remove(barc, remcols)
    linc = choice(remcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_88a62173():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 14))
    w = unifint(diff_lb, diff_ub, (1, 14))
    bgc = choice(cols)
    gib = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gib))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gib = fill(gib, col, chos)
        inds = difference(inds, chos)
    numchinv = unifint(diff_lb, diff_ub, (0, h * w - 1))
    numch = h * w - numchinv
    inds2 = totuple(asindices(gib))
    subs = sample(inds2, numch)
    go = hmirror(hmirror(gib))
    for x, y in subs:
        go = fill(go, choice(remove(go[x][y], colsch + [bgc])), {(x, y)})
    gi = canvas(bgc, (h*2+1, w*2+1))
    return gi


def generate_3aa6fb7a():
    diff_lb = 0
    diff_ub = 1
    base = (ORIGIN, RIGHT, DOWN, UNITY)
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_3ac3eb23():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nlocs = unifint(diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))
    locopts = interval(1, w - 1, 1)
    gi = canvas(bgc, (h, w))
    return gi


def generate_c3e719e8():
    diff_lb = 0
    diff_ub = 1
    cols = remove(0, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    gob = canvas(-1, (h**2, w**2))
    wg = canvas(-1, (h, w))
    ncols = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    nmc = randint(max(1, (h * w) // (ncols + 1) + 1), h * w)
    inds = totuple(asindices(wg))
    mc = choice(cols)
    remcols = remove(mc, cols)
    mcc = sample(inds, nmc)
    inds = difference(inds, mcc)
    gi = fill(wg, mc, mcc)
    return gi


def generate_29c11459():
    diff_lb = 0
    diff_ub = 1
    colopts = remove(5, interval(0, 10, 1))
    gi = canvas(0, (1, 1))
    return gi


def generate_23b5c85d():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    colopts = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_1bfc4729():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    if h % 2 == 1:
        h = choice((max(4, h - 1), min(30, h + 1)))
    alocj = unifint(diff_lb, diff_ub, (w // 2, w - 1))
    if choice((True, False)):
        alocj = max(min(w // 2, alocj - w // 2), 1)
    aloci = randint(1, h // 2 - 1)
    blocj = unifint(diff_lb, diff_ub, (w // 2, w - 1))
    if choice((True, False)):
        blocj = max(min(w // 2, blocj - w // 2), 1)
    bloci = randint(h // 2, h - 2)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_47c1f68c():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 14))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc, linc = sample(cols, 2)
    remcols = difference(cols, (bgc, linc))
    objc = choice(remcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w - 1))
    bx = asindices(canv)
    obj = {choice(totuple(bx))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, obj)
        ch = choice(totuple(bx & dns))
        obj.add(ch)
        bx = bx - {ch}
    obj = recolor(objc, obj)
    gi = paint(canv, obj)
    return gi


def generate_178fcbfb():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_ae4f1146():
    diff_lb = 0
    diff_ub = 1
    cols = remove(1, interval(0, 10, 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    dh = unifint(diff_lb, diff_ub, (2, h // 3))
    dw = unifint(diff_lb, diff_ub, (2, w // 3))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // (2 * dh * dw)))
    cards = interval(0, dh * dw, 1)
    ccards = sorted(sample(cards, min(num, len(cards))))
    sgs = []
    c1 = canvas(fgc, (dh, dw))
    inds = totuple(asindices(c1))
    for card in ccards:
        x = sample(inds, card)
        x1 = fill(c1, 1, x)
        sgs.append(asobject(x1))
    go = paint(c1, sgs[-1])
    gi = canvas(bgc, (h, w))
    return gi


def generate_3de23699():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    c = canvas(bgc, (h, w))
    hi = unifint(diff_lb, diff_ub, (4, h))
    wi = unifint(diff_lb, diff_ub, (4, w))
    loci = randint(0, h - hi)
    locj = randint(0, w - wi)
    remcols = remove(bgc, cols)
    ccol = choice(remcols)
    remcols = remove(ccol, remcols)
    ncol = choice(remcols)
    tmpo = frozenset({(loci, locj), (loci + hi - 1, locj + wi - 1)})
    cnds = totuple(backdrop(inbox(tmpo)))
    mp = len(cnds) // 2
    dev = unifint(diff_lb, diff_ub, (0, mp))
    ncnds = choice((dev, len(cnds) - dev))
    ncnds = min(max(0, ncnds), len(cnds))
    ss = sample(cnds, ncnds)
    gi = fill(c, ccol, corners(tmpo))
    return gi


def generate_7ddcd7ec():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    crns = (((0, 0), (-1, -1)), ((0, 1), (-1, 1)), ((1, 0), (1, -1)), ((1, 1), (1, 1)))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_5c2c9af4():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    boxhd = unifint(diff_lb, diff_ub, (0, h // 2))
    boxwd = unifint(diff_lb, diff_ub, (0, w // 2))
    boxh = choice((boxhd, h - boxhd))
    boxw = choice((boxwd, w - boxwd))
    if boxh % 2 == 0:
        boxh = choice((boxh - 1, boxh + 1))
    if boxw % 2 == 0:
        boxw = choice((boxw - 1, boxw + 1))
    boxh = min(max(1, boxh), h if h % 2 == 1 else h - 1)
    boxw = min(max(1, boxw), w if w % 2 == 1 else w - 1)
    boxshap = (boxh, boxw)
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)
    loc = (loci, locj)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    cpi = loci + boxh // 2
    cpj = locj + boxw // 2
    cp = (cpi, cpj)
    A = (loci, locj)
    B = (loci + boxh - 1, locj + boxw - 1)
    gi = fill(c, fgc, {A, B, cp})
    return gi


def generate_0b148d64():
    diff_lb = 0
    diff_ub = 1
    itv = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    bgc = choice(itv)
    remitv = remove(bgc, itv)
    g = canvas(bgc, (h, w))
    x = randint(3, h - 3)
    y = randint(3, w - 3)
    di = randint(2, h - x - 1)
    dj = randint(2, w - y - 1)
    A = backdrop(frozenset({(0, 0), (x, y)}))
    B = backdrop(frozenset({(x + di, 0), (h - 1, y)}))
    C = backdrop(frozenset({(0, y + dj), (x, w - 1)}))
    D = backdrop(frozenset({(x + di, y + dj), (h - 1, w - 1)}))
    cola = choice(remitv)
    colb = choice(remove(cola, remitv))
    trg = choice((A, B, C, D))
    rem = remove(trg, (A, B, C, D))
    subf = lambda bx: {
        choice(totuple(connect(ulcorner(bx), urcorner(bx)))),
        choice(totuple(connect(ulcorner(bx), llcorner(bx)))),
        choice(totuple(connect(urcorner(bx), lrcorner(bx)))),
        choice(totuple(connect(llcorner(bx), lrcorner(bx)))),
    }
    sampler = lambda bx: set(sample(
        totuple(bx),
        len(bx) - unifint(diff_lb, diff_ub, (0, len(bx) - 1))
    ))
    gi = fill(g, cola, sampler(trg) | subf(trg))
    return gi


def generate_beb8660c():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    w = unifint(diff_lb, diff_ub, (3, 30))
    h = unifint(diff_lb, diff_ub, (w, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_8d510a79():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    barloci = randint(2, h - 3)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_7468f01a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    sgc, fgc = sample(remcols, 2)
    oh = unifint(diff_lb, diff_ub, (2, max(2, int(h * (2/3)))))
    ow = unifint(diff_lb, diff_ub, (2, max(2, int(w * (2/3)))))
    gi = canvas(bgc, (h, w))
    return gi


def generate_09629e4f():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nrows, ncolumns = h, w
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    ncols = unifint(diff_lb, diff_ub, (2, min(7, (h * w) - 2)))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    fullh, fullw = h * nrows + nrows - 1, w * ncolumns + ncolumns - 1
    gi = canvas(barcol, (fullh, fullw))
    return gi


def generate_4347f46a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_6d58a25d():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    shp = normalize(frozenset({
    (0, 0), (1, 0), (1, 1), (1, -1), (2, -1), (2, -2), (2, 1), (2, 2), (3, 3), (3, -3)
    }))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    c1 = choice(remcols)
    c2 = choice(remove(c1, remcols))
    loci = randint(0, h - 4)
    locj = randint(0, w - 7)
    plcd = shift(shp, (loci, locj))
    rem = difference(inds, plcd)
    nnoise = unifint(diff_lb, diff_ub, (1, max(1, len(rem) // 2 - 1)))
    nois = sample(rem, nnoise)
    gi = fill(c, c2, nois)
    return gi


def generate_363442ee():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 3))
    w = unifint(diff_lb, diff_ub, (1, 3))
    h = h * 2 + 1
    w = w * 2 + 1
    nremh = unifint(diff_lb, diff_ub, (2, 30 // h))
    nremw = unifint(diff_lb, diff_ub, (2, (30 - w - 1) // w))
    rsh = nremh * h
    rsw = nremw * w
    rss = (rsh, rsw)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    rsi = canvas(bgc, rss)
    rso = canvas(bgc, rss)
    ls = canvas(bgc, ((nremh - 1) * h, w))
    ulc = canvas(bgc, (h, w))
    bar = canvas(barcol, (nremh * h, 1))
    dotcands = totuple(product(interval(0, rsh, h), interval(0, rsw, w)))
    dotcol = choice(remcols)
    dev = unifint(diff_lb, diff_ub, (1, len(dotcands) // 2))
    ndots = choice((dev, len(dotcands) - dev))
    ndots = min(max(1, ndots), len(dotcands))
    dots = sample(dotcands, ndots)
    nfullremcols = unifint(diff_lb, diff_ub, (1, 8))
    fullremcols = sample(remcols, nfullremcols)
    for ij in asindices(ulc):
        ulc = fill(ulc, choice(fullremcols), {ij})
    ulco = asobject(ulc)
    osf = (h//2, w//2)
    for d in dots:
        rsi = fill(rsi, dotcol, {add(osf, d)})
        rso = paint(rso, shift(ulco, d))
    gi = hconcat(hconcat(vconcat(ulc, ls), bar), rsi)
    return gi


def generate_855e0971():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    nbarsd = unifint(diff_lb, diff_ub, (1, 4))
    nbars = choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    h = unifint(diff_lb, diff_ub, (nbars, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    barsizes = [2] * nbars
    while sum(barsizes) < h:
        j = randint(0, nbars - 1)
        barsizes[j] += 1
    gi = tuple()
    return gi


def generate_137eaa0f():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 4))
    w = unifint(diff_lb, diff_ub, (2, 4))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    dotc = choice(remcols)
    remcols = remove(dotc, remcols)
    go = canvas(dotc, (h, w))
    inds = totuple(asindices(go))
    loc = choice(inds)
    reminds = remove(loc, inds)
    nc = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    choscols = sample(remcols, nc)
    cd = {c: set() for c in choscols}
    for c in choscols:
        ij = choice(reminds)
        cd[c].add(ij)
        reminds = remove(ij, reminds)
    for ri in reminds:
        cd[choice(choscols)].add(ri)
    for c, idxes in cd.items():
        go = fill(go, c, idxes)
    gih = unifint(diff_lb, diff_ub, (min(h, w) * 2, 30))
    giw = unifint(diff_lb, diff_ub, (min(h, w) * 2, 30))
    objs = tuple(
        normalize(insert((dotc, loc), frozenset({(c, ij) for ij in cd[c]}))) \
            for c in choscols
    )
    maxtr = min(h, w) * 2
    maxtrtot = 1000
    while True:
        succ = True
        gi = canvas(bgc, (gih, giw))
        return gi


def generate_31aa019c():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (5, 30))
        w = unifint(diff_lb, diff_ub, (5, 30))
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        canv = canvas(bgc, (h, w))
        inds = totuple(asindices(canv))
        mp = (h * w) // 2 - 1
        ncols = unifint(diff_lb, diff_ub, (2, min(9, mp // 2 - 1)))
        chcols = sample(cols, ncols)
        trgcol = chcols[0]
        chcols = chcols[1:]
        dic = {c: set() for c in chcols}
        nnoise = unifint(diff_lb, diff_ub, (2 * (ncols - 1), mp))
        locc = choice(inds)
        inds = remove(locc, inds)
        noise = sample(inds, nnoise)
        for c in chcols:
            ij = choice(inds)
            dic[c].add(ij)
            inds = remove(ij, inds)
        for c in chcols:
            ij = choice(inds)
            dic[c].add(ij)
            inds = remove(ij, inds)
        for ij in noise:
            c = choice(chcols)
            dic[c].add(ij)
            inds = remove(ij, inds)
        gi = fill(canv, trgcol, {locc})
        return gi


def generate_2bee17df():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    indord1 = apply(tojvec, interval(0, w, 1))
    indord2 = apply(rbind(astuple, w - 1), interval(1, h - 1, 1))
    indord3 = apply(lbind(astuple, h - 1), interval(w - 1, 0, -1))
    indord4 = apply(toivec, interval(h - 1, 0, -1))
    indord = indord1 + indord2 + indord3 + indord4
    k = len(indord)
    sp = randint(0, k)
    arr = indord[sp:] + indord[:sp]
    ep = randint(k // 2 - 3, k // 2 + 1)
    a = arr[:ep]
    b = arr[ep:]
    cola = choice(remcols)
    remcols = remove(cola, remcols)
    colb = choice(remcols)
    gi = fill(c, cola, a)
    return gi


def generate_50cb2852():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_662c240a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 7))
    ng = unifint(diff_lb, diff_ub, (2, 30 // d))
    nc = unifint(diff_lb, diff_ub, (2, min(9, d ** 2)))
    c = canvas(-1, (d, d))
    inds = totuple(asindices(c))
    tria = sfilter(inds, lambda ij: ij[1] >= ij[0])
    tcolset = sample(cols, nc)
    triaf = frozenset((choice(tcolset), ij) for ij in tria)
    triaf = triaf | dmirror(triaf)
    gik = paint(c, triaf)
    ndistinv = unifint(diff_lb, diff_ub, (0, (d * (d - 1) // 2 - 1)))
    ndist = d * (d - 1) // 2 - ndistinv
    distinds = sample(difference(inds, sfilter(inds, lambda ij: ij[0] == ij[1])), ndist)
    for ij in distinds:
        if gik[ij[0]][ij[1]] == gik[ij[1]][ij[0]]:
            gik = fill(gik, choice(remove(gik[ij[0]][ij[1]], tcolset)), {ij})
        else:
            gik = fill(gik, gik[ij[1]][ij[0]], {ij})
    gi = gik
    return gi


def generate_e8593010():
    diff_lb = 0
    diff_ub = 1
    a = frozenset({frozenset({ORIGIN})})
    b = frozenset({frozenset({ORIGIN, RIGHT}), frozenset({ORIGIN, DOWN})})
    c = frozenset({
    frozenset({ORIGIN, DOWN, UNITY}),
    frozenset({ORIGIN, DOWN, RIGHT}),
    frozenset({UNITY, DOWN, RIGHT}),
    frozenset({UNITY, ORIGIN, RIGHT}),
    shift(frozenset({ORIGIN, UP, DOWN}), DOWN),
    shift(frozenset({ORIGIN, LEFT, RIGHT}), RIGHT)
    })
    a, b, c = totuple(a), totuple(b), totuple(c)
    prs = [(a, 3), (b, 2), (c, 1)]
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_d9f24cd1():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    dotc = choice(remcols)
    locopts = interval(1, w - 1, 1)
    maxnloc = (w - 2) // 2
    nlins = unifint(diff_lb, diff_ub, (1, maxnloc))
    locs = []
    for k in range(nlins):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        locopts = remove(loc, locopts)
        locopts = remove(loc - 1, locopts)
        locopts = remove(loc + 1, locopts)
        locs.append(loc)
    ndots = unifint(diff_lb, diff_ub, (1, maxnloc))
    locopts = interval(1, w - 1, 1)
    dotlocs = []
    for k in range(ndots):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        locopts = remove(loc, locopts)
        locopts = remove(loc - 1, locopts)
        locopts = remove(loc + 1, locopts)
        dotlocs.append(loc)
    gi = canvas(bgc, (h, w))
    return gi


def generate_90c28cc7():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 10))
    w = unifint(diff_lb, diff_ub, (2, 10))
    nc = unifint(diff_lb, diff_ub, (2, 9))
    gi = canvas(-1, (h, w))
    return gi


def generate_321b1fc6():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    objh = unifint(diff_lb, diff_ub, (2, 5))
    objw = unifint(diff_lb, diff_ub, (2, 5))
    bounds = asindices(canvas(0, (objh, objw)))
    shp = {choice(totuple(bounds))}
    nc = unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for j in range(nc):
        ij = choice(totuple((bounds - shp) & mapply(dneighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    dmyc = choice(remcols)
    remcols = remove(dmyc, remcols)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    shpp = shift(shp, (loci, locj))
    numco = unifint(diff_lb, diff_ub, (2, 8))
    colll = sample(remcols, numco)
    shppc = frozenset({(choice(colll), ij) for ij in shpp})
    while numcolors(shppc) == 1:
        shppc = frozenset({(choice(colll), ij) for ij in shpp})
    shppcn = normalize(shppc)
    gi = canvas(bgc, (h, w))
    return gi


def generate_6455b5f5():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 8))
    while True:
        h = unifint(diff_lb, diff_ub, (6, 30))
        w = unifint(diff_lb, diff_ub, (6, 30))
        bgc = choice(cols)
        fgc = choice(remove(bgc, cols))
        gi = canvas(bgc, (h, w))
        return gi


def generate_4c5c2cf0():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(diff_lb, diff_ub, (2, (w - 3) // 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    cc = choice(remcols)
    remcols = remove(cc, remcols)
    objc = choice(remcols)
    sg = canvas(bgc, (oh, ow))
    locc = (oh - 1, ow - 1)
    sg = fill(sg, cc, {locc})
    reminds = totuple(remove(locc, asindices(sg)))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
    cells = sample(reminds, ncells)
    while ncells == 5 and shape(cells) == (3, 3):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
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
    canv = canvas(bgc, (h, w))
    loci = randint(0, h - 2 * oh - 1)
    locj = randint(0, w - 2 * ow - 1)
    loc = (loci, locj)
    go = paint(canv, shift(gg, loc))
    gi = paint(canv, shift(asobject(sg), loc))
    return gi


def generate_56ff96f3():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_2c608aff():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    boxh = unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccol = choice(remcols)
    remcols = remove(ccol, remcols)
    dcol = choice(remcols)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (h, w))
    return gi


def generate_e98196ab():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (3, 14))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    topc = choice(remcols)
    remcols = remove(topc, remcols)
    botc = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    nocc = unifint(diff_lb, diff_ub, (2, (h * w) // 2))
    subs = sample(inds, nocc)
    numa = randint(1, nocc - 1)
    A = sample(subs, numa)
    B = difference(subs, A)
    topg = fill(c, topc, A)
    botg = fill(c, botc, B)
    go = fill(topg, botc, B)
    br = canvas(linc, (1, w))
    gi = vconcat(vconcat(topg, br), botg)
    return gi


def generate_c9f8e694():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = 0
    remcols = remove(bgc, cols)
    sqc = choice(remcols)
    remcols = remove(sqc, remcols)
    ncols = unifint(diff_lb, diff_ub, (1, min(h, 8)))
    nsq = unifint(diff_lb, diff_ub, (1, 8))
    gir = canvas(bgc, (h, w - 1))
    gil = tuple((choice(remcols),) for j in range(h))
    inds = asindices(gir)
    succ = 0
    fails = 0
    maxfails = nsq * 5
    while succ < nsq and fails < maxfails:
        loci = randint(0, h - 3)
        locj = randint(0, w - 3)
        lock = randint(loci+1, min(loci + max(1, 2*h//3), h - 1))
        locl = randint(locj+1, min(locj + max(1, 2*w//3), w - 1))
        bd = backdrop(frozenset({(loci, locj), (lock, locl)}))
        if bd.issubset(inds):
            gir = fill(gir, sqc, bd)
            succ += 1
            indss = inds - bd
        else:
            fails += 1
    locs = ofcolor(gir, sqc)
    gil = tuple(e if idx in apply(first, locs) else (bgc,) for idx, e in enumerate(gil))
    fullobj = toobject(locs, hupscale(gil, w))
    gi = hconcat(gil, gir)
    return gi


def generate_eb5a1d5d():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 10))
    go = canvas(-1, (d*2-1, d*2-1))
    colss = sample(cols, d)
    for j, cc in enumerate(colss):
        go = fill(go, cc, box(frozenset({(j, j), (2*d - 2 - j, 2*d - 2 - j)})))
    nvenl = unifint(diff_lb, diff_ub, (0, 30 - d))
    nhenl = unifint(diff_lb, diff_ub, (0, 30 - d))
    enl = [nvenl, nhenl]
    gi = tuple(e for e in go)
    return gi


def generate_82819916():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ass, bss = sample(remcols, 2)
    itv = interval(0, w, 1)
    na = randint(2, w - 2)
    alocs = sample(itv, na)
    blocs = difference(itv, alocs)
    if min(alocs) > min(blocs):
        alocs, blocs = blocs, alocs
    llocs = randint(0, h - 1)
    gi = canvas(bgc, (h, w))
    return gi


def generate_5daaa586():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    loci1 = randint(1, h - 4)
    locj1 = randint(1, w - 4)
    loci1dev = unifint(diff_lb, diff_ub, (0, loci1 - 1))
    locj1dev = unifint(diff_lb, diff_ub, (0, locj1 - 1))
    loci1 -= loci1dev
    locj1 -= locj1dev
    loci2 = unifint(diff_lb, diff_ub, (loci1 + 2, h - 2))
    locj2 = unifint(diff_lb, diff_ub, (locj1 + 2, w - 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c1, c2, c3, c4 = sample(remcols, 4)
    f1 = recolor(c1, hfrontier(toivec(loci1)))
    f2 = recolor(c2, hfrontier(toivec(loci2)))
    f3 = recolor(c3, vfrontier(tojvec(locj1)))
    f4 = recolor(c4, vfrontier(tojvec(locj2)))
    gi = canvas(bgc, (h, w))
    return gi


def generate_68b16354():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_bb43febb():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_9ecd008a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    return gi


def generate_f25ffba3():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 14))
    h = h * 2 + 1
    w = unifint(diff_lb, diff_ub, (3, 15))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (2, h * w - 2))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    while uppermost(obj) > h // 2 - 1 or lowermost(obj) < h // 2 + 1:
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gix = paint(canv, obj)
    gix = apply(rbind(order, matcher(identity, bgc)), gix)
    gi = hconcat(gix, canv)
    return gi


def generate_3bdb4ada():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_2013d3e2():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 10))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    return gi


def generate_aabf363d():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 28))
    w = unifint(diff_lb, diff_ub, (3, 28))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    cola = choice(remcols)
    remcols = remove(cola, remcols)
    colb = choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    sp = choice(totuple(bounds))
    ub = min(h * w - 1, max(1, (2/3) * h * w))
    ncells = unifint(diff_lb, diff_ub, (1, ub))
    shp = {sp}
    for k in range(ncells):
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = shift(shp, (1, 1))
    c2 = canvas(bgc, (h+2, w+2))
    gi = fill(c2, cola, shp)
    return gi


def generate_d037b0a7():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_e26a3af2():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    nr = unifint(diff_lb, diff_ub, (1, 10))
    w = unifint(diff_lb, diff_ub, (4, 30))
    scols = sample(cols, nr)
    sgs = [canvas(col, (2, w)) for col in scols]
    numexp = unifint(diff_lb, diff_ub, (0, 30 - nr))
    for k in range(numexp):
        idx = randint(0, nr - 1)
        sgs[idx] = sgs[idx] + sgs[idx][-1:]
    sgs2 = []
    for idx, col in enumerate(scols):
        sg = sgs[idx]
        a, b = shape(sg)
        ub = (a * b) // 2 - 1
        nnoise = unifint(diff_lb, diff_ub, (0, ub))
        inds = totuple(asindices(sg))
        noise = sample(inds, nnoise)
        oc = remove(col, cols)
        noise = frozenset({(choice(oc), ij) for ij in noise})
        sg2 = paint(sg, noise)
        for idxx in [0, -1]:
            while sum([e == col for e in sg2[idxx]]) < w // 2:
                locs = [j for j, e in enumerate(sg2[idxx]) if e != col]
                ch = choice(locs)
                if idxx == 0:
                    sg2 = (sg2[0][:ch] + (col,) + sg2[0][ch+1:],) + sg2[1:]
                else:
                    sg2 = sg2[:-1] + (sg2[-1][:ch] + (col,) + sg2[-1][ch+1:],)
        sgs2.append(sg2)
    gi = tuple(row for sg in sgs2 for row in sg)
    return gi


def generate_b8825c91():
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    return gi


def generate_ba97ae07():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_c909285e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    nfronts = unifint(diff_lb, diff_ub, (1, (h + w) // 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    boxcol = choice(remcols)
    remcols = remove(boxcol, remcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_d511f180():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (5, 8))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    numbg = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    bginds = sample(inds, numbg)
    idx = randint(0, numbg)
    blues = bginds[:idx]
    greys = bginds[idx:]
    rem = difference(inds, bginds)
    gi = fill(c, 8, blues)
    return gi


def generate_d0f5fe59():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, min(30, (h * w) // 9)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    nfound = 0
    trials = 0
    maxtrials = nobjs * 5
    gi = canvas(bgc, (h, w))
    return gi


def generate_6e82a1ae():
    diff_lb = 0
    diff_ub = 1
    b = frozenset({frozenset({ORIGIN, RIGHT}), frozenset({ORIGIN, DOWN})})
    c = frozenset({
    frozenset({ORIGIN, DOWN, UNITY}),
    frozenset({ORIGIN, DOWN, RIGHT}),
    frozenset({UNITY, DOWN, RIGHT}),
    frozenset({UNITY, ORIGIN, RIGHT}),
    shift(frozenset({ORIGIN, UP, DOWN}), DOWN),
    shift(frozenset({ORIGIN, LEFT, RIGHT}), RIGHT)
    })
    d = set()
    for k in range(100):
        shp = {(0, 0)}
        for jj in range(3):
            shp.add(choice(totuple(mapply(dneighbors, shp) - shp)))
        shp = frozenset(normalize(shp))
        d.add(shp)
    d = frozenset(d)
    d, b, c = totuple(d), totuple(b), totuple(c)
    prs = [(b, 3), (c, 2), (d, 1)]
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_f2829549():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(hconcat(A, bar), B)
    return gi


def generate_ce22a75a():
    diff_lb = 0
    diff_ub = 1
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    ndots = unifint(diff_lb, diff_ub, (1, (h * w) // 3))
    dots = sample(totuple(asindices(c)), ndots)
    gi = fill(c, fgc, dots)
    return gi


def generate_3c9b0459():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_99b1bc43():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(hconcat(A, bar), B)
    return gi


def generate_b6afb2da():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2, 4))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_c8f0f002():
    diff_lb = 0
    diff_ub = 1
    cols = remove(7, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    numo = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    orng = sample(inds, numo)
    rem = difference(inds, orng)
    gi = fill(c, 7, orng)
    return gi


def generate_54d82841():
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nshps = unifint(diff_lb, diff_ub, (1, w // 3))
    gi = canvas(bgc, (h, w))
    return gi


def generate_d631b094():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc = 0
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    nc = unifint(diff_lb, diff_ub, (1, min(30, (h * w) // 2 - 1)))
    c = canvas(bgc, (h, w))
    cands = totuple(asindices(c))
    cels = sample(cands, nc)
    gi = fill(c, fgc, cels)
    return gi


def generate_7c008303():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 13))
    w = unifint(diff_lb, diff_ub, (2, 13))
    h = h * 2
    w = w * 2
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    fgc = choice(remcols)
    remcols = remove(fgc, remcols)
    fremcols = sample(remcols, unifint(diff_lb, diff_ub, (1, 4)))
    qc = [choice(fremcols) for j in range(4)]
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    ncd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = choice((ncd, h * w - ncd))
    nc = min(max(0, nc), h * w)
    cels = sample(inds, nc)
    go = fill(c, fgc, cels)
    gi = canvas(bgc, (h + 3, w + 3))
    return gi


def generate_dae9d2b5():
    diff_lb = 0
    diff_ub = 1
    cols = remove(6, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    if len(set(aset) & set(bset)) == 0:
        bset = bset[:-1] + [choice(aset)]
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(A, B)
    return gi


def generate_aedd82e4():
    diff_lb = 0
    diff_ub = 1
    colopts = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = 0
    remcols = remove(bgc, colopts)
    c = canvas(bgc, (h, w))
    card_bounds = (0, max(0, (h * w) // 2 - 1))
    num = unifint(diff_lb, diff_ub, card_bounds)
    numcols = unifint(diff_lb, diff_ub, (0, min(8, num)))
    inds = totuple(asindices(c))
    chosinds = sample(inds, num)
    choscols = sample(remcols, numcols)
    locs = interval(0, len(chosinds), 1)
    choslocs = sample(locs, numcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_c9e6f938():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_913fb3ed():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (1, 30)
    cols = difference(interval(0, 10, 1), (1, 2, 3, 4, 6, 8))
    sr = (2, 3, 8)
    tr = (1, 6, 4)
    prs = list(zip(sr, tr))
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_6430c8c4():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(hconcat(A, bar), B)
    return gi


def generate_c0f76784():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (6, 7, 8))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, len(remcols)))
    ccols = sample(remcols, numcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_3af2c5a8():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_496994bd():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = asindices(canv)
    obj = {
        (choice(remcols), choice(totuple(sfilter(bx, lambda ij: ij[0] < h//2)))),
        (choice(remcols), choice(totuple(sfilter(bx, lambda ij: ij[0] > h//2))))
    }
    for kk in range(nc - 2):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gix = paint(canv, obj)
    gix = apply(rbind(order, matcher(identity, bgc)), gix)
    flag = choice((True, False))
    gi = hconcat(gix, canv if flag else hconcat(canvas(bgc, (h, 1)), canv))
    return gi


def generate_bd4472b8():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 28))
    w = unifint(diff_lb, diff_ub, (2, 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    ccols = sample(remcols, w)
    cc = (tuple(ccols),)
    br = canvas(linc, (1, w))
    lp = canvas(bgc, (h, w))
    gi = vconcat(vconcat(cc, br), lp)
    return gi


def generate_fafffa47():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(A, B)
    return gi


def generate_67e8384a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 14))
    w = unifint(diff_lb, diff_ub, (1, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    return gi


def generate_ed36ccf7():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_67a3c6ac():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_a416b8f3():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_d10ecb37():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_5bd6f4ac():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_7b7f7511():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 15))
    bgc = choice(cols)
    go = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, min(9, h * w - 1)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(go))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        go = fill(go, col, chos)
        inds = difference(inds, chos)
    if choice((True, False)):
        go = dmirror(go)
        gi = vconcat(go, go)
        return gi


def generate_c59eb873():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_b1948b0a():
    diff_lb = 0
    diff_ub = 1
    cols = remove(6, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    npd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    np = choice((npd, h * w - npd))
    np = min(max(0, npd), h * w)
    gi = canvas(6, (h, w))
    return gi


def generate_25ff71a9():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    nc = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc-1):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    return gi


def generate_f25fbde4():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    ncd = unifint(diff_lb, diff_ub, (1, max(1, (min(15, h-1) * min(15, w-1)) // 2)))
    nc = choice((ncd, (h-1) * (w-1) - ncd))
    nc = min(max(1, ncd), (h-1) * (w-1) - 1)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(canvas(-1, (min(15, h - 1), min(15, w - 1))))
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    return gi


def generate_a740d043():
    diff_lb = 0
    diff_ub = 1
    cols = remove(0, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    ncd = unifint(diff_lb, diff_ub, (1, max(1, ((h-1) * (w-1)) // 2)))
    nc = choice((ncd, (h-1) * (w-1) - ncd))
    nc = min(max(1, ncd), (h-1) * (w-1) - 1)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, len(remcols)))
    remcols = sample(remcols, numc)
    c = canvas(bgc, (h, w))
    bounds = asindices(canvas(-1, (h - 1, w - 1)))
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    obj = {(choice(remcols), ij) for ij in plcd}
    gi = paint(c, obj)
    return gi


def generate_be94b721():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    no = unifint(diff_lb, diff_ub, (3, max(3, (h * w) // 16)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (no+1, max(no+1, 2*no)))
    inds = asindices(c)
    ch = choice(totuple(inds))
    shp = {ch}
    inds = remove(ch, inds)
    for k in range(nc - 1):
        shp.add(choice(totuple((inds - shp) & mapply(dneighbors, shp))))
    inds = (inds - shp) - mapply(neighbors, shp)
    trgc = choice(remcols)
    gi = fill(c, trgc, shp)
    return gi


def generate_44d8ac46():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_3618c87e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, linc, dotc = sample(cols, 3)
    c = canvas(bgc, (h, w))
    ln = connect((0, 0), (0, w - 1))
    nlocs = unifint(diff_lb, diff_ub, (1, w//2))
    locs = []
    opts = interval(0, w, 1)
    for k in range(nlocs):
        if len(opts) == 0:
            break
        ch = choice(opts)
        locs.append(ch)
        opts = remove(ch, opts)
        opts = remove(ch-1, opts)
        opts = remove(ch+1, opts)
    nlocs = len(opts)
    gi = fill(c, linc, ln)
    return gi


def generate_b27ca6d3():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, dotc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    gi = canvas(bgc, (h, w))
    return gi


def generate_46f33fce():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 7))
    w = unifint(diff_lb, diff_ub, (2, 7))
    nc = unifint(diff_lb, diff_ub, (0, (h * w) // 2 - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    go = canvas(bgc, (h, w))
    gi = canvas(bgc, (h*2, w*2))
    return gi


def generate_a79310a0():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    nc = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc - 1):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    return gi


def generate_dc1df850():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (0, (h * w) // 2 - 1))
    nreddev = unifint(diff_lb, diff_ub, (0, nc // 2))
    nred = choice((nreddev, nc - nreddev))
    nred = min(max(0, nred), nc)
    inds = totuple(asindices(c))
    occ = sample(inds, nc)
    reds = sample(occ, nred)
    others = difference(occ, reds)
    c = fill(c, 2, reds)
    obj = frozenset({(choice(remcols), ij) for ij in others})
    c = paint(c, obj)
    gi = tuple(r for r in c)
    return gi


def generate_f76d97a5():
    diff_lb = 0
    diff_ub = 1
    cols = remove(0, remove(5, interval(0, 10, 1)))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    col = choice(cols)
    gi = canvas(5, (h, w))
    return gi


def generate_0d3d703e():
    diff_lb = 0
    diff_ub = 1
    incols = (1, 2, 3, 4, 5, 6, 8, 9)
    outcols = (5, 6, 4, 3, 1, 2, 9, 8)
    k = len(incols)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    gi = canvas(-1, (h, w))
    return gi


def generate_445eab21():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_b94a9452():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, outer, inner = sample(cols, 3)
    c = canvas(bgc, (h, w))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    oh2d = unifint(diff_lb, diff_ub, (0, oh // 2))
    ow2d = unifint(diff_lb, diff_ub, (0, ow // 2))
    oh2 = choice((oh2d, oh - oh2d))
    oh2 = min(max(1, oh2), oh - 2)
    ow2 = choice((ow2d, ow - ow2d))
    ow2 = min(max(1, ow2), ow - 2)
    loci2 = randint(loci+1, loci+oh-oh2-1)
    locj2 = randint(locj+1, locj+ow-ow2-1)
    obj1 = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    obj2 = backdrop(frozenset({(loci2, locj2), (loci2 + oh2 - 1, locj2 + ow2 - 1)}))
    gi = fill(c, outer, obj1)
    return gi


def generate_e9afcf9a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    numc = unifint(diff_lb, diff_ub, (1, min(10, h)))
    colss = sample(cols, numc)
    rr = tuple(choice(colss) for k in range(h))
    rr2 = rr[::-1]
    gi = []
    return gi


def generate_e9614598():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    r = randint(0, h - 1)
    sizh = unifint(diff_lb, diff_ub, (2, w//2))
    siz = 2 * sizh + 1
    siz = min(max(5, siz), w)
    locj = randint(0, w - siz)
    bgc, dotc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    A = (r, locj)
    B = (r, locj+siz-1)
    gi = fill(c, dotc, {A, B})
    return gi


def generate_d23f8c26():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (2, 30))
    wh = unifint(diff_lb, diff_ub, (1, 14))
    w = 2 * wh + 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_ce9e57f2():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    nbars = unifint(diff_lb, diff_ub, (2, (w - 2) // 2))
    locopts = interval(1, w - 1, 1)
    barlocs = []
    for k in range(nbars):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        barlocs.append(loc)
        locopts = remove(loc, locopts)
        locopts = remove(loc + 1, locopts)
        locopts = remove(loc - 1, locopts)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 8))
    colss = sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    return gi


def generate_b9b7f026():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_6d75e8bb():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    nc = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc - 1):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    return gi


def generate_3f7978a0():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, noisec, linec = sample(cols, 3)
    c = canvas(bgc, (h, w))
    oh = unifint(diff_lb, diff_ub, (4, max(4, int((2/3) * h))))
    oh = min(oh, h)
    ow = unifint(diff_lb, diff_ub, (4, max(4, int((2/3) * w))))
    ow = min(ow, w)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    nnoise = unifint(diff_lb, diff_ub, (0, (h * w) // 4))
    inds = totuple(asindices(c))
    noise = sample(inds, nnoise)
    gi = fill(c, noisec, noise)
    return gi


def generate_e76a88a6():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    objh = unifint(diff_lb, diff_ub, (2, 5))
    objw = unifint(diff_lb, diff_ub, (2, 5))
    bounds = asindices(canvas(0, (objh, objw)))
    shp = {choice(totuple(bounds))}
    nc = unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for j in range(nc):
        ij = choice(totuple((bounds - shp) & mapply(dneighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    dmyc = choice(remcols)
    remcols = remove(dmyc, remcols)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    shpp = shift(shp, (loci, locj))
    numco = unifint(diff_lb, diff_ub, (2, 8))
    colll = sample(remcols, numco)
    shppc = frozenset({(choice(colll), ij) for ij in shpp})
    while numcolors(shppc) == 1:
        shppc = frozenset({(choice(colll), ij) for ij in shpp})
    shppcn = normalize(shppc)
    gi = canvas(bgc, (h, w))
    return gi


def generate_a61f2674():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, remove(1, interval(0, 10, 1)))
    w = unifint(diff_lb, diff_ub, (5, 28))
    h = unifint(diff_lb, diff_ub, (w // 2 + 1, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_ce4f8723():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    cola = choice(remcols)
    colb = choice(remove(cola, remcols))
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    return gi


def generate_caa06a1f():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    vp = unifint(diff_lb, diff_ub, (2, h//2-1))
    hp = unifint(diff_lb, diff_ub, (2, w//2-1))
    bgc = choice(cols)
    numc = unifint(diff_lb, diff_ub, (2, min(8, max(2, hp * vp))))
    remcols = remove(bgc, cols)
    ccols = sample(remcols, numc)
    remcols = difference(remcols, ccols)
    tric = choice(remcols)
    obj = {(choice(ccols), ij) for ij in asindices(canvas(-1, (vp, hp)))}
    go = canvas(bgc, (h, w))
    gi = canvas(bgc, (h, w))
    return gi


def generate_94f9d214():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(A, B)
    return gi


def generate_feca6190():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    w = unifint(diff_lb, diff_ub, (2, 6))
    bgc = 0
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, min(w, 5)))
    ccols = sample(remcols, ncols)
    cands = interval(0, w, 1)
    locs = sample(cands, ncols)
    gi = canvas(bgc, (1, w))
    return gi


def generate_d5d6de2d():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_4612dd53():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    ih = unifint(diff_lb, diff_ub, (5, h-1))
    iw = unifint(diff_lb, diff_ub, (5, w-1))
    bgc, col = sample(cols, 2)
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    bx = box(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    if choice((True, False)):
        locc = randint(loci + 2, loci + ih - 3)
        br = connect((locc, locj+1), (locc, locj + iw - 2))
    else:
        locc = randint(locj + 2, locj + iw - 3)
        br = connect((loci+1, locc), (loci + ih - 2, locc))
    c = canvas(bgc, (h, w))
    crns = sample(totuple(corners(bx)), 3)
    onbx = totuple(crns)
    rembx = difference(bx, crns)
    onbr = sample(totuple(br), 2)
    rembr = difference(br, onbr)
    noccbx = unifint(diff_lb, diff_ub, (0, len(rembx)))
    noccbr = unifint(diff_lb, diff_ub, (0, len(rembr)))
    occbx = sample(totuple(rembx), noccbx)
    occbr = sample(totuple(rembr), noccbr)
    c = fill(c, col, bx)
    c = fill(c, col, br)
    gi = fill(c, bgc, occbx)
    return gi


def generate_1f642eb9():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    ih = unifint(diff_lb, diff_ub, (2, min(h - 4, 2 * (h // 3))))
    iw = unifint(diff_lb, diff_ub, (2, min(w - 4, 2 * (w // 3))))
    loci = randint(2, h - ih - 2)
    locj = randint(2, w - iw - 2)
    bgc, sqc = sample(cols, 2)
    remcols = difference(cols, (bgc, sqc))
    numcells = unifint(diff_lb, diff_ub, (1, 2 * ih + 2 * iw - 4))
    outs = []
    ins = []
    c1 = choice((True, False))
    c2 = choice((True, False))
    c3 = choice((True, False))
    c4 = choice((True, False))
    for a in range(loci + (not c1), loci + ih - (not c2)):
        outs.append((a, 0))
        ins.append((a, locj))
    for a in range(loci + (not c3), loci + ih - (not c4)):
        outs.append((a, w - 1))
        ins.append((a, locj + iw - 1))
    for b in range(locj + c1, locj + iw - (c3)):
        outs.append((0, b))
        ins.append((loci, b))
    for b in range(locj + (c2), locj + iw - (c4)):
        outs.append((h - 1, b))
        ins.append((loci + ih - 1, b))
    inds = interval(0, 2 * ih + 2 * iw - 4, 1)
    locs = sample(inds, numcells)
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    outs = [e for j, e in enumerate(outs) if j in locs]
    ins = [e for j, e in enumerate(ins) if j in locs]
    c = canvas(bgc, (h, w))
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    gi = fill(c, sqc, bd)
    return gi


def generate_681b3aeb():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    fullsuc = False
    while not fullsuc:
        hi = unifint(diff_lb, diff_ub, (2, 8))
        wi = unifint(diff_lb, diff_ub, (2, 8))
        h = unifint(diff_lb, diff_ub, ((3*hi, 30)))
        w = unifint(diff_lb, diff_ub, ((3*wi, 30)))
        c = canvas(-1, (hi, hi))
        bgc, ca, cb = sample(cols, 3)
        gi = canvas(bgc, (h, w))
        return gi


def generate_d364b489():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 6, 7, 8))    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_25d8a9c8():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    gi = []
    return gi


def generate_bda2d7a6():
    diff_lb = 0
    diff_ub = 1
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 14))
    w = unifint(diff_lb, diff_ub, (2, 14))
    ncols = unifint(diff_lb, diff_ub, (2, 10))
    cols = sample(colopts, ncols)
    colord = [choice(cols) for j in range(min(h, w))]
    shp = (h*2, w*2)
    gi = canvas(0, shp)
    return gi


def generate_a5f85a15():
    diff_lb = 0
    diff_ub = 1
    colopts = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    startlocs = apply(toivec, interval(h - 1, 0, -1)) + apply(tojvec, interval(0, w, 1))
    cands = interval(0, h + w - 1, 1)
    num = unifint(diff_lb, diff_ub, (1, (h + w - 1) // 3))
    locs = []
    for k in range(num):
        if len(cands) == 0:
            break
        loc = choice(cands)
        locs.append(loc)
        cands = remove(loc, cands)
        cands = remove(loc - 1, cands)
        cands = remove(loc + 1, cands)
    locs = set([startlocs[loc] for loc in locs])
    bgc, fgc = sample(colopts, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_32597951():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    ih = unifint(diff_lb, diff_ub, (2, h // 2))
    iw = unifint(diff_lb, diff_ub, (2, w // 2))
    bgc, noisec, fgc = sample(cols, 3)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    ndev = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    num = choice((ndev, h * w - ndev))
    num = min(max(num, 0), h * w)
    ofc = sample(inds, num)
    c = fill(c, noisec, ofc)
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    tofillfc = bd & ofcolor(c, bgc)
    gi = fill(c, fgc, tofillfc)
    return gi


def generate_cf98881b():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 9))
    bgc, barcol, cola, colb, colc = sample(cols, 5)
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    devc = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    sgnc = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    devc = sgnc * devc
    numa = mp + deva
    numb = mp + devb
    numc = mp + devc
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    numc = max(min(h * w - 1, numc), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    c = sample(inds, numc)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gic = fill(canv, colc, c)
    gi = hconcat(hconcat(hconcat(gia, gbar), hconcat(gib, gbar)), gic)
    return gi


def generate_41e4d17e():
    diff_lb = 0
    diff_ub = 1
    cols = remove(6, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_91714a58():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    bgc, targc = sample(cols, 2)
    remcols = remove(bgc, cols)
    nnoise = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    gi = canvas(bgc, (h, w))
    return gi


def generate_b60334d2():
    diff_lb = 0
    diff_ub = 1
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 9))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_952a094c():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    ih = unifint(diff_lb, diff_ub, (4, h - 2))
    iw = unifint(diff_lb, diff_ub, (4, w - 2))
    loci = randint(1, h - ih - 1)
    locj = randint(1, w - iw - 1)
    sp = (loci, locj)
    ep = (loci + ih - 1, locj + iw - 1)
    bx = box(frozenset({sp, ep}))
    bgc, fgc, a, b, c, d = sample(cols, 6)
    canv = canvas(bgc, (h, w))
    canvv = fill(canv, fgc, bx)
    gi = tuple(e for e in canvv)
    return gi


def generate_b8cdaf2b():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc, linc, dotc = sample(cols, 3)
    lin = connect((0, 0), (0, w - 1))
    winv = unifint(diff_lb, diff_ub, (2, w - 1))
    w2 = w - winv
    w2 = min(max(w2, 1), w - 2)
    locj = randint(1, w - w2 - 1)
    bar2 = connect((0, locj), (0, locj + w2 - 1))
    c = canvas(bgc, (h, w))
    gi = fill(c, linc, lin)
    return gi


def generate_b548a754():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    hi = unifint(diff_lb, diff_ub, (4, h - 1))
    wi = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - hi)
    locj = randint(0, w - wi)
    bx = box(frozenset({(loci, locj), (loci + hi - 1, locj + wi - 1)}))
    ins = backdrop(inbox(bx))
    bgc, boxc, inc, dotc = sample(cols, 4)
    c = canvas(bgc, (h, w))
    go = fill(c, boxc, bx)
    go = fill(go, inc, ins)
    cutoff = randint(loci + 2, loci + hi - 2)
    bx2 = box(frozenset({(loci, locj), (cutoff, locj + wi - 1)}))
    ins2 = backdrop(inbox(bx2))
    gi = fill(c, boxc, bx2)
    return gi


def generate_95990924():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2, 3, 4))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_f1cefba8():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    ih = unifint(diff_lb, diff_ub, (6, h - 1))
    iw = unifint(diff_lb, diff_ub, (6, w - 1))
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    bgc, ringc, inc = sample(cols, 3)
    obj = frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)})
    ring1 = box(obj)
    ring2 = inbox(obj)
    bd = backdrop(obj)
    c = canvas(bgc, (h, w))
    c = fill(c, inc, bd)
    c = fill(c, ringc, ring1 | ring2)
    cands = totuple(ring2 - corners(ring2))
    numc = unifint(diff_lb, diff_ub, (1, len(cands) // 2))
    locs = sample(cands, numc)
    gi = fill(c, inc, locs)
    return gi


def generate_c444b776():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 9))
    w = unifint(diff_lb, diff_ub, (2, 9))
    nh = unifint(diff_lb, diff_ub, (1, 3))
    nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
    bgclinc = sample(cols, 2)
    bgc, linc = bgclinc
    remcols = difference(cols, bgclinc)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    inds = totuple(asindices(smallc))
    numcol = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numcol)
    numcels = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    cels = sample(inds, numcels)
    obj = {(choice(ccols), ij) for ij in cels}
    smallcpainted = paint(smallc, obj)
    llocs = set()
    for a in range(0, fullh, h+1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    srcloc = choice(llocs)
    obj = asobject(smallcpainted)
    gi = paint(c, shift(obj, srcloc))
    return gi


def generate_97999447():
    diff_lb = 0
    diff_ub = 1
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    opts = interval(0, h, 1)
    num = unifint(diff_lb, diff_ub, (1, h))
    locs = sample(opts, num)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    return gi


def generate_d89b689b():
    diff_lb = 0
    diff_ub = 1
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, sqc, a, b, c, d = sample(cols, 6)
    loci = randint(1, h - 3)
    locj = randint(1, w - 3)
    canv = canvas(bgc, (h, w))
    go = fill(canv, a, {(loci, locj)})
    go = fill(go, b, {(loci, locj+1)})
    go = fill(go, c, {(loci+1, locj)})
    go = fill(go, d, {(loci+1, locj+1)})
    inds = totuple(asindices(canv))
    aopts = sfilter(inds, lambda ij: ij[0] < loci and ij[1] < locj)
    bopts = sfilter(inds, lambda ij: ij[0] < loci and ij[1] > locj + 1)
    copts = sfilter(inds, lambda ij: ij[0] > loci + 1 and ij[1] < locj)
    dopts = sfilter(inds, lambda ij: ij[0] > loci + 1 and ij[1] > locj + 1)
    aopts = order(aopts, lambda ij: manhattan({ij}, {(loci, locj)}))
    bopts = order(bopts, lambda ij: manhattan({ij}, {(loci, locj + 1)}))
    copts = order(copts, lambda ij: manhattan({ij}, {(loci + 1, locj)}))
    dopts = order(dopts, lambda ij: manhattan({ij}, {(loci + 1, locj + 1)}))
    aidx = unifint(diff_lb, diff_ub, (0, len(aopts) - 1))
    bidx = unifint(diff_lb, diff_ub, (0, len(bopts) - 1))
    cidx = unifint(diff_lb, diff_ub, (0, len(copts) - 1))
    didx = unifint(diff_lb, diff_ub, (0, len(dopts) - 1))
    loca = aopts[aidx]
    locb = bopts[bidx]
    locc = copts[cidx]
    locd = dopts[didx]
    gi = fill(canv, sqc, backdrop({(loci, locj), (loci + 1, locj + 1)}))
    return gi


def generate_543a7ed5():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (3, 4))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    return gi


def generate_a2fd1cf0():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 3, 8))    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    gloci = unifint(diff_lb, diff_ub, (1, h - 1))
    glocj = unifint(diff_lb, diff_ub, (1, w - 1))
    gloc = (gloci, glocj)
    bgc = choice(cols)
    g = canvas(bgc, (h, w))
    g = fill(g, 3, {gloc})
    g = rot180(g)
    glocinv = center(ofcolor(g, 3))
    glocinvi, glocinvj = glocinv
    rloci = unifint(diff_lb, diff_ub, (glocinvi+1, h - 1))
    rlocj = unifint(diff_lb, diff_ub, (glocinvj+1, w - 1))
    rlocinv = (rloci, rlocj)
    g = fill(g, 2, {rlocinv})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(g)
    return gi


def generate_cdecee7f():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    numc = unifint(diff_lb, diff_ub, (1, min(9, w)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    inds = interval(0, w, 1)
    locs = sample(inds, numc)
    locs = order(locs, identity)
    gi = canvas(bgc, (h, w))
    return gi


def generate_0962bcdd():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (3, 4))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (2, 7))
    ccols = sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    return gi


def generate_dc0a314f():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    return gi


def generate_29623171():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 6))
    w = unifint(diff_lb, diff_ub, (2, 6))
    nh = unifint(diff_lb, diff_ub, (2, 4))
    nw = unifint(diff_lb, diff_ub, (2, 4))
    bgc, linc, fgc = sample(cols, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    inds = totuple(asindices(smallc))
    llocs = set()
    for a in range(0, fullh, h+1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    srcloc = choice(llocs)
    nmostc = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    mostc = sample(inds, nmostc)
    srcg = fill(smallc, fgc, mostc)
    obj = asobject(srcg)
    shftd = shift(obj, srcloc)
    gi = paint(c, shftd)
    return gi


def generate_d4a91cb9():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 4, 8))    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    gloci = unifint(diff_lb, diff_ub, (1, h - 1))
    glocj = unifint(diff_lb, diff_ub, (1, w - 1))
    gloc = (gloci, glocj)
    bgc = choice(cols)
    g = canvas(bgc, (h, w))
    g = fill(g, 8, {gloc})
    g = rot180(g)
    glocinv = center(ofcolor(g, 8))
    glocinvi, glocinvj = glocinv
    rloci = unifint(diff_lb, diff_ub, (glocinvi+1, h - 1))
    rlocj = unifint(diff_lb, diff_ub, (glocinvj+1, w - 1))
    rlocinv = (rloci, rlocj)
    g = fill(g, 2, {rlocinv})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(g)
    return gi


def generate_60b61512():
    diff_lb = 0
    diff_ub = 1
    cols = remove(7, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_4938f0c2():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 31))
    w = unifint(diff_lb, diff_ub, (10, 31))
    oh = unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(diff_lb, diff_ub, (2, (w - 3) // 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    cc = choice(remcols)
    remcols = remove(cc, remcols)
    objc = choice(remcols)
    sg = canvas(bgc, (oh, ow))
    locc = (oh - 1, ow - 1)
    sg = fill(sg, cc, {locc})
    reminds = totuple(remove(locc, asindices(sg)))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
    cells = sample(reminds, ncells)
    while ncells == 4 and shape(cells) == (2, 2):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
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
    canv = canvas(bgc, (h, w))
    loci = randint(0, h - 2 * oh - 1)
    locj = randint(0, w - 2 * ow - 1)
    loc = (loci, locj)
    go = paint(canv, shift(gg, loc))
    gi = paint(canv, shift(asobject(sg), loc))
    return gi


def generate_a8d7556c():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (0, 2))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    fgc = choice(cols)
    c = canvas(fgc, (h, w))
    numblacks = unifint(diff_lb, diff_ub, (1, (h * w) // 3 * 2))
    inds = totuple(asindices(c))
    blacks = sample(inds, numblacks)
    gi = fill(c, 0, blacks)
    return gi


def generate_007bbfb7():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    c = canvas(0, (h, w))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcd, h * w - numcd))
    numc = min(max(1, numc), h * w - 1)
    inds = totuple(asindices(c))
    locs = sample(inds, numc)
    fgc = choice(cols)
    gi = fill(c, fgc, locs)
    return gi


def generate_b190f7f5():
    diff_lb = 0
    diff_ub = 1
    fullcols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    bgc = choice(fullcols)
    cols = remove(bgc, fullcols)
    c = canvas(bgc, (h, w))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcd, h * w - numcd))
    numc = min(max(1, numc), h * w - 1)
    numcd2 = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc2 = choice((numcd2, h * w - numcd2))
    numc2 = min(max(2, numc2), h * w - 1)
    inds = totuple(asindices(c))
    srclocs = sample(inds, numc)
    srccol = choice(cols)
    remcols = remove(srccol, cols)
    numcols = unifint(diff_lb, diff_ub, (2, 8))
    trglocs = sample(inds, numc2)
    ccols = sample(remcols, numcols)
    fixc1 = choice(ccols)
    trgobj = [(fixc1, trglocs[0]), (choice(remove(fixc1, ccols)), trglocs[1])] + [(choice(ccols), ij) for ij in trglocs[2:]]
    trgobj = frozenset(trgobj)
    gisrc = fill(c, srccol, srclocs)
    gitrg = paint(c, trgobj)
    catf = choice((hconcat, vconcat))
    ordd = choice(([gisrc, gitrg], [gitrg, gisrc]))
    gi = catf(*ordd)
    return gi


def generate_2bcee788():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 20))
    w = unifint(diff_lb, diff_ub, (2, 10))
    bgc, sepc, objc = sample(cols, 3)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    spi = randint(0, h - 1)
    sp = (spi, w - 1)
    shp = {sp}
    numcellsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcellsd, h * w - numcellsd))
    numc = min(max(2, numc), h * w - 1)
    reminds = set(remove(sp, inds))
    for k in range(numc):
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    while width(shp) == 1:
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    c2 = fill(c, objc, shp)
    borderinds = sfilter(shp, lambda ij: ij[1] == w - 1)
    c3 = fill(c, sepc, borderinds)
    gimini = asobject(hconcat(c2, vmirror(c3)))
    gomini = asobject(hconcat(c2, vmirror(c2)))
    fullh = unifint(diff_lb, diff_ub, (h+1, 30))
    fullw = unifint(diff_lb, diff_ub, (2*w+1, 30))
    fullg = canvas(bgc, (fullh, fullw))
    loci = randint(0, fullh - h)
    locj = randint(0, fullw - 2 * w)
    loc = (loci, locj)
    gi = paint(fullg, gimini)
    return gi


def generate_a3df8b1e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    w = unifint(diff_lb, diff_ub, (2, 10))
    h = unifint(diff_lb, diff_ub, (w+1, 30))
    bgc, linc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    return gi


def generate_80af3007():
    diff_lb = 0
    diff_ub = 1
    fullcols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    bgc = choice(fullcols)
    cols = remove(bgc, fullcols)
    c = canvas(bgc, (h, w))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcd, h * w - numcd))
    numc = min(max(0, numc), h * w)
    inds = totuple(asindices(c))
    locs = tuple(set(sample(inds, numc)) | set(sample(totuple(corners(inds)), 3)))
    fgc = choice(cols)
    gi = fill(c, fgc, locs)
    return gi


def generate_e50d258f():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    padcol = choice(remcols)
    remcols = remove(padcol, remcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_0e206a2e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, acol, bcol, ccol, Dcol = sample(cols, 5)
    gi = canvas(bgc, (h, w))
    return gi


def generate_b230c067():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2))
    while True:
        h = unifint(diff_lb, diff_ub, (10, 30))
        w = unifint(diff_lb, diff_ub, (10, 30))
        oh = unifint(diff_lb, diff_ub, (2, h // 3 - 1))
        ow = unifint(diff_lb, diff_ub, (2, w // 3 - 1))
        numcd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        numc = choice((numcd, oh * ow - numcd))
        numca = min(max(2, numc), oh * ow - 2)
        bounds = asindices(canvas(-1, (oh, ow)))
        sp = choice(totuple(bounds))
        shp = {sp}
        for k in range(numca):
            ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
            shp.add(ij)
        shpa = normalize(shp)
        shpb = set(normalize(shp))
        mxnch = oh * ow - len(shpa)
        nchinv = unifint(diff_lb, diff_ub, (1, mxnch))
        nch = mxnch - nchinv
        nch = min(max(1, nch), mxnch)
        for k in range(nch):
            ij = choice(totuple((bounds - shpb) & mapply(neighbors, shpb)))
            shpb.add(ij)
        if choice((True, False)):
            shpa, shpb = shpb, shpa
        bgc, fgc = sample(cols, 2)
        c = canvas(bgc, (h, w))
        inds = asindices(c)
        acands = sfilter(inds, lambda ij: ij[0] <= h - height(shpa) and ij[1] <= w - width(shpa))
        aloc = choice(totuple(acands))
        aplcd = shift(shpa, aloc)
        gi = fill(c, fgc, aplcd)
        return gi


def generate_db93a21d():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 3))
    h = unifint(diff_lb, diff_ub, (12, 31))
    w = unifint(diff_lb, diff_ub, (12, 32))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_1e32b0e9():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 6))
    w = unifint(diff_lb, diff_ub, (4, 6))
    nh = unifint(diff_lb, diff_ub, (1, 4))
    nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
    bgc, linc, fgc = sample(cols, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    llocs = set()
    for a in range(0, fullh, h+1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    srcloc = choice(llocs)
    remlocs = remove(srcloc, llocs)
    ncells = unifint(diff_lb, diff_ub, (0, (h - 2) * (w - 2) - 1))
    smallc2 = canvas(bgc, (h-2, w - 2))
    inds = asindices(smallc2)
    sp = choice(totuple(inds))
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(ncells):
        ij = choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = shift(shp, (1, 1))
    gg = asobject(fill(smallc, fgc, shp))
    gg2 = asobject(fill(smallc, linc, shp))
    gi = paint(c, shift(gg, srcloc))
    return gi


def generate_6773b310():
    diff_lb = 0
    diff_ub = 1
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh = unifint(diff_lb, diff_ub, (2, 5))
    nw = unifint(diff_lb, diff_ub, (2, 5))
    bgc, linc, fgc = sample(cols, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    llocs = set()
    for a in range(0, fullh, h + 1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    nbldev = unifint(diff_lb, diff_ub, (0, (nh * nw) // 2))
    nbl = choice((nbldev, nh * nw - nbldev))
    nbl = min(max(1, nbl), nh * nw - 1)
    bluelocs = sample(llocs, nbl)
    bglocs = difference(llocs, bluelocs)
    inds = totuple(asindices(smallc))
    gi = tuple(e for e in c)
    return gi


def generate_6ecd11f4():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 7))
    w = unifint(diff_lb, diff_ub, (2, 7))
    bgc, fgc = sample(cols, 2)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    inds = asindices(canvas(bgc, (h, w)))
    nlocsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nlocs = choice((nlocsd, h * w - nlocsd))
    nlocs = min(max(3, nlocs), h * w - 1)
    sp = choice(totuple(inds))
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(nlocs):
        ij = choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    h, w = shape(shp)
    canv = canvas(bgc, (h, w))
    objbase = fill(canv, fgc, shp)
    maxhscf = (2*h+h+1) // h
    maxwscf = (2*w+w+1) // w
    hscf = unifint(diff_lb, diff_ub, (2, maxhscf))
    wscf = unifint(diff_lb, diff_ub, (2, maxwscf))
    obj = asobject(hupscale(vupscale(objbase, hscf), wscf))
    oh, ow = shape(obj)
    inds = asindices(canv)
    objx = {(choice(ccols), ij) for ij in inds}
    if len(palette(objx)) == 1:
        objxodo = first(objx)
        objx = insert((choice(remove(objxodo[0], ccols)), objxodo[1]), remove(objxodo, objx))
    fullh = unifint(diff_lb, diff_ub, (hscf*h+h+1, 30))
    fullw = unifint(diff_lb, diff_ub, (wscf*w+w+1, 30))
    gi = canvas(bgc, (fullh, fullw))
    return gi


def generate_8403a5d5():
    diff_lb = 0
    diff_ub = 1
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    loccinv = unifint(diff_lb, diff_ub, (1, w - 1))
    locc = w - loccinv
    bgc, fgc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    idx = (h - 1, locc)
    gi = fill(c, fgc, {idx})
    return gi


def generate_941d9a10():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    opts = interval(2, (h-1)//2 + 1, 2)
    nhidx = unifint(diff_lb, diff_ub, (0, len(opts) - 1))
    nh = opts[nhidx]
    opts = interval(2, (w-1)//2 + 1, 2)
    nwidx = unifint(diff_lb, diff_ub, (0, len(opts) - 1))
    nw = opts[nwidx]
    bgc, fgc = sample(cols, 2)
    hgrid = canvas(bgc, (2*nh+1, w))
    for j in range(1, h, 2):
        hgrid = fill(hgrid, fgc, connect((j, 0), (j, w)))
    for k in range(h - (2*nh+1)):
        loc = randint(0, height(hgrid) - 1)
        hgrid = hgrid[:loc] + canvas(bgc, (1, w)) + hgrid[loc:]
    wgrid = canvas(bgc, (2*nw+1, h))
    for j in range(1, w, 2):
        wgrid = fill(wgrid, fgc, connect((j, 0), (j, h)))
    for k in range(w - (2*nw+1)):
        loc = randint(0, height(wgrid) - 1)
        wgrid = wgrid[:loc] + canvas(bgc, (1, h)) + wgrid[loc:]
    wgrid = dmirror(wgrid)
    gi = canvas(bgc, (h, w))
    return gi


def generate_b0c4d837():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    bgc, boxc, fillc = sample(cols, 3)
    subg = canvas(boxc, (oh, ow))
    subg2 = canvas(fillc, (oh-1, ow-2))
    ntofill = unifint(diff_lb, diff_ub, (1, min(9, oh-2)))
    for j in range(ntofill):
        subg2 = fill(subg2, bgc, connect((j, 0), (j, ow-2)))
    subg = paint(subg, shift(asobject(subg2), (0, 1)))
    gi = canvas(bgc, (h, w))
    return gi


def generate_0a938d79():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 29))
    w = unifint(diff_lb, diff_ub, (h+1, 30))
    bgc, cola, colb = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_b7249182():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    ih = unifint(diff_lb, diff_ub, (3, (h-1)//2))
    bgc, ca, cb = sample(cols, 3)
    subg = canvas(bgc, (ih, 5))
    gi = canvas(bgc, (h, w))
    return gi


def generate_7b6016b9():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, remove(2, interval(0, 10, 1)))
    while True:
        h = unifint(diff_lb, diff_ub, (5, 30))
        w = unifint(diff_lb, diff_ub, (5, 30))
        bgc, fgc = sample(cols, 2)
        numl = unifint(diff_lb, diff_ub, (4, min(h, w)))
        gi = canvas(bgc, (h, w))
        return gi


def generate_72ca375d():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    srcobjh = unifint(diff_lb, diff_ub, (2, 8))
    srcobjwh = unifint(diff_lb, diff_ub, (1, 4))
    bnds = asindices(canvas(-1, (srcobjh, srcobjwh)))
    spi = randint(0, srcobjh - 1)
    sp = (spi, srcobjwh - 1)
    srcobj = {sp}
    bnds = remove(sp, bnds)
    ncellsd = unifint(diff_lb, diff_ub, (0, (srcobjh * srcobjwh) // 2))
    ncells1 = choice((ncellsd, srcobjh * srcobjwh - ncellsd))
    ncells2 = unifint(diff_lb, diff_ub, (1, srcobjh * srcobjwh))
    ncells = (ncells1 + ncells2) // 2
    ncells = min(max(1, ncells), srcobjh * srcobjwh, (h * w) // 2 - 1)
    for k in range(ncells - 1):
        srcobj.add(choice(totuple((bnds - srcobj) & mapply(neighbors, srcobj))))
    srcobj = normalize(srcobj)
    srcobj = srcobj | shift(vmirror(srcobj), (0, width(srcobj)))
    srcobjh, srcobjw = shape(srcobj)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    trgc = choice(remcols)
    go = canvas(bgc, (srcobjh, srcobjw))
    go = fill(go, trgc, srcobj)
    loci = randint(0, h - srcobjh)
    locj = randint(0, w - srcobjw)
    locc = (loci, locj)
    gi = canvas(bgc, (h, w))
    return gi


def generate_673ef223():
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    barh = unifint(diff_lb, diff_ub, (2, (h-1)//2))
    ncells = unifint(diff_lb, diff_ub, (1, barh))
    bgc, barc, dotc = sample(cols, 3)
    sg = canvas(bgc, (barh, w))
    topsgi = fill(sg, barc, connect((0, 0), (barh-1, 0)))
    return gi


def generate_868de0fa():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 7))    
    h = unifint(diff_lb, diff_ub, (9, 30))
    w = unifint(diff_lb, diff_ub, (9, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_40853293():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    nlines = unifint(diff_lb, diff_ub, (2, min(8, (h*w)//2)))
    nhorilines = randint(1, nlines - 1)
    nvertilines = nlines - nhorilines
    ilocs = interval(0, h, 1)
    ilocs = sample(ilocs, min(nhorilines, len(ilocs)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_6e19193c():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    dirs = (
    ((0, 0), (-1, -1)),
    ((0, 1), (-1, 1)),
    ((1, 0), (1, -1)),
    ((1, 1), (1, 1))
    )
    base = ((0, 0), (1, 0), (0, 1), (1, 1))
    candsi = [
    set(base) - {dr[0]} for dr in dirs
    ]
    candso = [
    (set(base) | shoot(dr[0], dr[1])) - {dr[0]} for dr in dirs
    ]
    cands = list(zip(candsi, candso))    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_8731374e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    inh = randint(5, h - 2)
    inw = randint(5, w - 2)
    bgc, fgc = sample(cols, 2)
    num = unifint(diff_lb, diff_ub, (1, min(inh, inw)))
    mat = canvas(bgc, (inh - 2, inw - 2))
    tol = lambda g: list(list(e) for e in g)
    tot = lambda g: tuple(tuple(e) for e in g)
    mat = fill(mat, fgc, connect((0, 0), (num - 1, num - 1)))
    mat = tol(mat)
    shuffle(mat)
    mat = tol(dmirror(tot(mat)))
    shuffle(mat)
    mat = dmirror(tot(mat))
    sgi = paint(canvas(bgc, (inh, inw)), shift(asobject(mat), (1, 1)))
    return gi


def generate_cce03e0d():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 8))    
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nred = unifint(diff_lb, diff_ub, (1, h * w - 1))
    ncols = unifint(diff_lb, diff_ub, (1, min(8, nred)))
    ncells = unifint(diff_lb, diff_ub, (1, h * w - nred))
    ccols = sample(cols, ncols)
    gi = canvas(0, (h, w))
    return gi


def generate_f9012d9b():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)    
    hp = unifint(diff_lb, diff_ub, (2, 10))
    wp = unifint(diff_lb, diff_ub, (2, 10))
    srco = canvas(0, (hp, wp))
    inds = asindices(srco)
    nc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, nc)
    obj = {(choice(ccols), ij) for ij in inds}
    srco = paint(srco, obj)
    gi = paint(srco, obj)
    return gi


def generate_f8ff0b80():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, min(30, (h * w) // 25)))
    gi = canvas(bgc, (h, w))
    return gi


def generate_e21d9049():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    ph = unifint(diff_lb, diff_ub, (2, 9))
    pw = unifint(diff_lb, diff_ub, (2, 9))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    hbar = frozenset({(choice(remcols), (k, 0)) for k in range(ph)})
    wbar = frozenset({(choice(remcols), (0, k)) for k in range(pw)})
    locih = randint(0, h - ph)
    locjh = randint(0, w - 1)
    loch = (locih, locjh)
    locjw = randint(0, w - pw)
    lociw = randint(0, h - 1)
    locw = (lociw, locjw)
    canv = canvas(bgc, (h, w))
    hbar = shift(hbar, loch)
    wbar = shift(wbar, locw)
    cp = (lociw, locjh)
    col = choice(remcols)
    hbard = extract(hbar, lambda cij: abs(cij[1][0] - lociw) % ph == 0)[1]
    hbar = sfilter(hbar, lambda cij: abs(cij[1][0] - lociw) % ph != 0) | {(col, hbard)}
    wbard = extract(wbar, lambda cij: abs(cij[1][1] - locjh) % pw == 0)[1]
    wbar = sfilter(wbar, lambda cij: abs(cij[1][1] - locjh) % pw != 0) | {(col, wbard)}
    gi = paint(canv, hbar | wbar)
    return gi


def generate_d4f3cd78():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    ih = unifint(diff_lb, diff_ub, (3, h//3*2))
    iw = unifint(diff_lb, diff_ub, (3, w//3*2))
    loci = randint(1, h - ih - 1)
    locj = randint(1, w - iw - 1)
    crns = frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)})
    fullcrns = corners(crns)
    bx = box(crns)
    opts = bx - fullcrns
    bgc, fgc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    nholes = unifint(diff_lb, diff_ub, (1, len(opts)))
    holes = sample(totuple(opts), nholes)
    gi = fill(c, fgc, bx - set(holes))
    return gi


def generate_9d9215db():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (5, 14))
    w = unifint(diff_lb, diff_ub, (5, 14))
    h = h * 2 + 1
    w = w * 2 + 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ub = min(h, w)//4
    nrings = unifint(diff_lb, diff_ub, (1, ub))
    onlinesbase = tuple([(2*k+1, 2*k+1) for k in range(ub)])
    onlines = sample(onlinesbase, nrings)
    onlines = {(choice(remcols), ij) for ij in onlines}
    gi = canvas(bgc, (h, w))
    return gi


def generate_0ca9ddb6():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2, 4, 6, 7, 8))
    xi = {(8, (0, 0))}
    xo = {(8, (0, 0))}
    ai = {(6, (0, 0))}
    ao = {(6, (0, 0))}
    bi = {(2, (1, 1))}
    bo = {(2, (1, 1))} | recolor(4, ineighbors((1, 1)))
    ci = {(1, (1, 1))}
    co = {(1, (1, 1))} | recolor(7, dneighbors((1, 1)))
    arr = ((ai, ao), (bi, bo), (ci, co), (xi, xo))    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 4))
    maxtr = 5 * nobjs
    tr = 0
    succ = 0
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_5521c0d9():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    inds = interval(0, w, 1)
    nobjs = unifint(diff_lb, diff_ub, (1, w//3))
    speps = sample(inds, nobjs*2)
    while 0 in speps or w - 1 in speps:
        nobjs = unifint(diff_lb, diff_ub, (1, w//3))
        speps = sample(inds, nobjs*2)
    speps = sorted(speps)
    starts = speps[::2]
    ends = speps[1::2]
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    forb = -1
    gi = canvas(bgc, (h, w))
    return gi


def generate_e3497940():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (3, 14))
    bgc, barc = sample(cols, 2)
    remcols = remove(barc, remove(bgc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    nlinesocc = unifint(diff_lb, diff_ub, (1, h))
    lopts = interval(0, h, 1)
    linesocc = sample(lopts, nlinesocc)
    rs = canvas(bgc, (h, w))
    ls = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for idx in linesocc:
        j = unifint(diff_lb, diff_ub, (1, w - 1))
        obj = [(choice(ccols), (idx, jj)) for jj in range(j)]
        go = paint(go, obj)
        slen = randint(1, j)
        obj2 = obj[:slen]
        if choice((True, False)):
            obj, obj2 = obj2, obj
        rs = paint(rs, obj)
        ls = paint(ls, obj2)
    gi = hconcat(hconcat(vmirror(ls), canvas(barc, (h, 1))), rs)
    return gi


def generate_6cdd2623():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    nnoisecols = unifint(diff_lb, diff_ub, (1, 7))
    noisecols = sample(remcols, nnoisecols)
    c = canvas(bgc, (h, w))
    ininds = totuple(shift(asindices(canvas(-1, (h-2, w-1))), (1, 1)))
    fixinds = sample(ininds, nnoisecols)
    fixobj = {(col, ij) for col, ij in zip(list(noisecols), fixinds)}
    gi = canvas(bgc, (h, w))
    return gi


def generate_dc433765():
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, src = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_d2abd087():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(difference(cols, (1, 2)))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_88a10436():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    objh = unifint(diff_lb, diff_ub, (0, 2))
    objw = unifint(diff_lb, diff_ub, (0 if objh > 0 else 1, 2))
    objh = objh * 2 + 1
    objw = objw * 2 + 1
    bb = asindices(canvas(-1, (objh, objw)))
    sp = (objh // 2, objw // 2)
    obj = {sp}
    bb = remove(sp, bb)
    ncells = unifint(diff_lb, diff_ub, (max(objh, objw), objh * objw))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    while height(obj) != objh or width(obj) != objw:
        obj.add(choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    bgc, fgc = sample(cols, 2)
    remcols = remove(bgc, remove(fgc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    obj = {(choice(ccols), ij) for ij in obj}
    obj = normalize(obj)
    gi = canvas(bgc, (h, w))
    return gi


def generate_05f2a901():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    objh = unifint(diff_lb, diff_ub, (2, min(w//2, h//2)))
    objw = unifint(diff_lb, diff_ub, (objh, w//2))
    bb = asindices(canvas(-1, (objh, objw)))
    sp = choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncells = unifint(diff_lb, diff_ub, (objh + objw, objh * objw))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    if height(obj) * width(obj) == len(obj):
        obj = remove(choice(totuple(obj)), obj)
    obj = normalize(obj)
    objh, objw = shape(obj)
    loci = unifint(diff_lb, diff_ub, (3, h - objh))
    locj = unifint(diff_lb, diff_ub, (0, w - objw))
    loc = (loci, locj)
    bgc, fgc, destc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_928ad970():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    ih = unifint(diff_lb, diff_ub, (9, h))
    iw = unifint(diff_lb, diff_ub, (9, w))
    bgc, linc, dotc = sample(cols, 3)
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    ulc = (loci, locj)
    lrc = (loci + ih - 1, locj + iw - 1)
    dot1 = choice(totuple(connect(ulc, (loci + ih - 1, locj)) - {ulc, (loci + ih - 1, locj)}))
    dot2 = choice(totuple(connect(ulc, (loci, locj + iw - 1)) - {ulc, (loci, locj + iw - 1)}))
    dot3 = choice(totuple(connect(lrc, (loci + ih - 1, locj)) - {lrc, (loci + ih - 1, locj)}))
    dot4 = choice(totuple(connect(lrc, (loci, locj + iw - 1)) - {lrc, (loci, locj + iw - 1)}))
    a, b = sorted(sample(interval(loci + 2, loci + ih - 2, 1), 2))
    while a + 1 == b:
        a, b = sorted(sample(interval(loci + 2, loci + ih - 2, 1), 2))
    c, d = sorted(sample(interval(locj + 2, locj + iw - 2, 1), 2))
    while c + 1 == d:
        c, d = sorted(sample(interval(locj + 2, locj + iw - 2, 1), 2))
    sp = box(frozenset({(a, c), (b, d)}))
    bx = {dot1, dot2, dot3, dot4}
    gi = canvas(bgc, (h, w))
    return gi


def generate_f8b3ba0a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 5))
    w = unifint(diff_lb, diff_ub, (1, 5))
    nh = unifint(diff_lb, diff_ub, (3, 29 // (h + 1)))
    nw = unifint(diff_lb, diff_ub, (3, 29 // (w + 1)))
    fullh = (h + 1) * nh + 1
    fullw = (w + 1) * nw + 1
    fullbgc, bgc = sample(cols, 2)
    remcols = remove(fullbgc, remove(bgc, cols))
    shp = shift(asindices(canvas(-1, (h, w))), (1, 1))
    gi = canvas(fullbgc, (fullh, fullw))
    return gi


def generate_fcb5c309():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, dotc, sqc = sample(cols, 3)
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    gi = canvas(bgc, (h, w))
    return gi


def generate_54d9e175():
    diff_lb = 0
    diff_ub = 1
    cols = (0, 5)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh = unifint(diff_lb, diff_ub, (1, 31 // (h + 1)))
    nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 31 // (w + 1)))
    fullh = (h + 1) * nh - 1
    fullw = (w + 1) * nw - 1
    linc, bgc = sample(cols, 2)
    gi = canvas(linc, (fullh, fullw))
    return gi


def generate_7f4411dc():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, fgc = sample(cols, 2)
    nsq = unifint(diff_lb, diff_ub, (1, (h * w) // 15))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    while tr < maxtr and succ < nsq:
        oh = randint(2, 6)
        ow = randint(2, 6)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        loci, locj = loc
        obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        obj = shift(obj, loc)
        if obj.issubset(inds):
            go = fill(go, fgc, obj)
            succ += 1
            inds = (inds - obj) - outbox(obj)
        tr += 1
    inds = ofcolor(go, bgc)
    nnoise = unifint(diff_lb, diff_ub, (0, len(inds) // 2 - 1))
    gi = tuple(e for e in go)
    return gi


def generate_67385a82():
    diff_lb = 0
    diff_ub = 1
    cols = remove(0, remove(8, interval(0, 10, 1)))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    col = choice(cols)
    gi = canvas(0, (h, w))
    return gi


def generate_d6ad076f():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    inh = unifint(diff_lb, diff_ub, (3, h))
    inw = unifint(diff_lb, diff_ub, (3, w))
    bgc, c1, c2 = sample(cols, 3)
    itv = interval(0, inh, 1)
    loci2i = unifint(diff_lb, diff_ub, (2, inh - 1))
    loci2 = itv[loci2i]
    itv = itv[:loci2i-1][::-1]
    loci1i = unifint(diff_lb, diff_ub, (0, len(itv) - 1))
    loci1 = itv[loci1i]
    cp = randint(1, inw - 2)
    ajs = randint(0, cp - 1)
    aje = randint(cp + 1, inw - 1)
    bjs = randint(0, cp - 1)
    bje = randint(cp + 1, inw - 1)
    obja = backdrop(frozenset({(0, ajs), (loci1, aje)}))
    objb = backdrop(frozenset({(loci2, bjs), (inh - 1, bje)}))
    c = canvas(bgc, (inh, inw))
    c = fill(c, c1, obja)
    c = fill(c, c2, objb)
    obj = asobject(c)
    loci = randint(0, h - inh)
    locj = randint(0, w - inw)
    loc = (loci, locj)
    obj = shift(obj, loc)
    gi = canvas(bgc, (h, w))
    return gi


def generate_e48d4e1a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    loci = randint(1, h - 2)
    locj = randint(1, w - 2)
    inds = asindices(canvas(-1, (loci, locj)))
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    nn = unifint(diff_lb, diff_ub, (1, maxn))
    ss = sample(totuple(inds), nn)
    bgc, fgc, dotc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_a48eeaf7():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    ih = unifint(diff_lb, diff_ub, (2, h//2))
    iw = unifint(diff_lb, diff_ub, (2, w//2))
    loci = randint(2, h - ih - 2)
    locj = randint(2, w - iw - 2)
    bgc, sqc, dotc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_56dc2b01():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 8))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    oh = unifint(diff_lb, diff_ub, (1, h))
    ow = unifint(diff_lb, diff_ub, (1, (w - 1) // 2 - 1))
    bb = asindices(canvas(-1, (oh, ow)))
    sp = choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    for k in range(ncells):
        obj.add(choice(totuple((bb - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    loci = randint(0, h - oh)
    locj = unifint(diff_lb, diff_ub, (1, w - ow))
    bgc, objc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_1caeab9d():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1,))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    oh = unifint(diff_lb, diff_ub, (1, h//2))
    ow = unifint(diff_lb, diff_ub, (1, w//3))
    bb = asindices(canvas(-1, (oh, ow)))
    sp = choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    for k in range(ncells):
        obj.add(choice(totuple((bb - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    loci = randint(0, h - oh)
    numo = unifint(diff_lb, diff_ub, (2, min(8, w // ow))) - 1
    itv = interval(0, w, 1)
    locj = randint(0, w - ow)
    objp = shift(obj, (loci, locj))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    gi = fill(c, 1, objp)
    return gi


def generate_b91ae062():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    numc = unifint(diff_lb, diff_ub, (3, min(h * w, min(10, 30 // max(h, w)))))
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    fixinds = sample(inds, numc)
    obj = {(cc, ij) for cc, ij in zip(ccols, fixinds)}
    for ij in difference(inds, fixinds):
        obj.add((choice(ccols), ij))
    gi = paint(c, obj)
    return gi


def generate_834ec97d():
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    loci = unifint(diff_lb, diff_ub, (0, h - 2))
    locjd = unifint(diff_lb, diff_ub, (0, w // 2))
    locj = choice((locjd, w - locjd))
    locj = min(max(0, locj), w - 1)
    loc = (loci, locj)
    bgc, fgc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    gi = fill(c, fgc, {loc})
    return gi


def generate_a699fb00():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    numls = unifint(diff_lb, diff_ub, (1, h - 1))
    opts = interval(0, h, 1)
    locs = sample(opts, numls)
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_91413438():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    maxnb = min(h * w - 1, min(30//h, 30//w))
    minnb = int(0.5 * ((4 * h * w + 1) ** 0.5 - 1)) + 1
    nbi = unifint(diff_lb, diff_ub, (0, maxnb - minnb))
    nb = min(max(minnb, maxnb - nbi), maxnb)
    fgc = choice(cols)
    c = canvas(0, (h, w))
    obj = sample(totuple(asindices(c)), h * w - nb)
    gi = fill(c, fgc, obj)
    return gi


def generate_99fa7670():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    num = unifint(diff_lb, diff_ub, (1, h // 2))
    inds = interval(0, h, 1)
    starts = sorted(sample(inds, num))
    ends = [x - 1 for x in starts[1:]] + [h - 1]
    nc = unifint(diff_lb, diff_ub, (1, 9))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, nc)
    gi = canvas(bgc, (h, w))
    return gi


def generate_d13f3404():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = unifint(diff_lb, diff_ub, (3, 15))
    vopts = {(ii, 0) for ii in interval(0, h, 1)}
    hopts = {(0, jj) for jj in interval(1, w, 1)}
    opts = tuple(vopts | hopts)
    num = unifint(diff_lb, diff_ub, (1, len(opts)))
    locs = sample(opts, num)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_c3f564a4():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    p = unifint(diff_lb, diff_ub, (2, min(9, min(h//3, w//3))))
    fixc = choice(cols)
    remcols = remove(fixc, cols)
    ccols = list(sample(remcols, p))
    shuffle(ccols)
    c = canvas(-1, (h, w))
    baseobj = {(cc, (0, jj)) for cc, jj in zip(ccols, range(p))}
    obj = {c for c in baseobj}
    while rightmost(obj) < 2 * max(w, h):
        obj = obj | shift(obj, (0, p))
    if choice((True, False)):
        obj = mapply(lbind(shift, obj), {(jj, 0) for jj in interval(0, h, 1)})
    else:
        obj = mapply(lbind(shift, obj), {(jj, -jj) for jj in interval(0, h, 1)})
    go = paint(c, obj)
    gi = tuple(e for e in go)
    return gi


def generate_ecdecbb3():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, dotc, linc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_ac0a08a4():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    num = unifint(diff_lb, diff_ub, (1, min(min(9, h * w - 2), min(30//h, 30//w))))
    bgc = choice(cols)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    locs = sample(totuple(inds), num)
    remcols = remove(bgc, cols)
    obj = {(col, loc) for col, loc in zip(sample(remcols, num), locs)}
    gi = paint(c, obj)
    return gi


def generate_22168020():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    num = unifint(diff_lb, diff_ub, (1, min(9, (h * w) // 10)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_ff805c23():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    return gi


def generate_4093f84a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    loci1, loci2 = sorted(sample(interval(2, h - 2, 1), 2))
    bgc, barc, dotc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_760b3cac():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    objL = frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)})
    objR = vmirror(objL)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (3, 14))
    w = 2 * w + 1
    bgc, objc, indc = sample(cols, 3)
    objh = unifint(diff_lb, diff_ub, (1, h - 3))
    objw = unifint(diff_lb, diff_ub, (1, w // 6))
    objw = 2 * objw + 1
    c = canvas(-1, (objh, objw))
    gi = canvas(bgc, (h, w))
    return gi


def generate_8efcae92():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, sqc, dotc = sample(cols, 3)
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    succ = 0
    maxtr = 4 * num
    tr = 0
    gi = canvas(bgc, (h, w))
    return gi


def generate_48d8fb45():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (2, (h * w) // 15))
    tr = 0
    maxtr = 4 * nobjs
    done = False
    succ = 0
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_8e1813be():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    bgc, sqc = sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    nbars = unifint(diff_lb, diff_ub, (3, 8))
    ccols = sample(remcols, nbars)
    w = unifint(diff_lb, diff_ub, (nbars+3, 30))
    hmarg = unifint(diff_lb, diff_ub, (2 * nbars, 30 - nbars))
    ccols = list(ccols)
    go = tuple(repeat(col, nbars) for col in ccols)
    gi = tuple(repeat(col, w) for col in ccols)
    return gi


def generate_5117e062():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (2, (h * w) // 15))
    tr = 0
    maxtr = 4 * nobjs
    done = False
    succ = 0
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_f15e1fac():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    nsps = unifint(diff_lb, diff_ub, (1, (w-1) // 2))
    ngps = unifint(diff_lb, diff_ub, (1, (h-1) // 2))
    spsj = sorted(sample(interval(1, w - 1, 1), nsps))
    gpsi = sorted(sample(interval(1, h - 1, 1), ngps))
    ofs = 0
    bgc, linc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_3906de3d():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    oh = unifint(diff_lb, diff_ub, (2, h // 2))
    ow = unifint(diff_lb, diff_ub, (3, w - 2))
    bgc, boxc, linc = sample(cols, 3)
    locj = randint(1, w - ow - 1)
    bx = backdrop(frozenset({(0, locj), (oh - 1, locj + ow - 1)}))
    gi = canvas(bgc, (h, w))
    return gi


def generate_77fdfe62():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 13))
    w = unifint(diff_lb, diff_ub, (1, 13))
    c1, c2, c3, c4, barc, bgc, inc = sample(cols, 7)
    qd = canvas(bgc, (h, w))
    inds = totuple(asindices(qd))
    fullh = 2 * h + 4
    fullw = 2 * w + 4
    n1 = unifint(diff_lb, diff_ub, (1, h * w))
    n2 = unifint(diff_lb, diff_ub, (1, h * w))
    n3 = unifint(diff_lb, diff_ub, (1, h * w))
    n4 = unifint(diff_lb, diff_ub, (1, h * w))
    i1 = sample(inds, n1)
    i2 = sample(inds, n2)
    i3 = sample(inds, n3)
    i4 = sample(inds, n4)
    gi = canvas(bgc, (2 * h + 4, 2 * w + 4))
    return gi


def generate_d406998b():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc, dotc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_694f12f3():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(diff_lb, diff_ub, (9, 30))
    w = unifint(diff_lb, diff_ub, (9, 30))
    seploc = randint(4, h - 5)
    bigh = unifint(diff_lb, diff_ub, (4, seploc))
    bigw = unifint(diff_lb, diff_ub, (3, w - 1))
    bigloci = randint(0, seploc - bigh)
    biglocj = randint(0, w - bigw)
    smallmaxh = h - seploc - 1
    smallmaxw = w - 1
    cands = []
    bigsize = bigh * bigw
    for a in range(3, smallmaxh+1):
        for b in range(3, smallmaxw+1):
            if a * b < bigsize:
                cands.append((a, b))
    cands = sorted(cands, key=lambda ab: ab[0]*ab[1])
    num = len(cands)
    idx = unifint(diff_lb, diff_ub, (0, num - 1))
    smallh, smallw = cands[idx]
    smallloci = randint(seploc+1, h - smallh)
    smalllocj = randint(0, w - smallw)
    bgc, sqc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_3befdf3e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numcols)
    nobjs = unifint(diff_lb, diff_ub, (1, ((h * w) // 40)))
    succ = 0
    maxtr = 5 * nobjs
    tr = 0
    gi = canvas(bgc, (h, w))
    return gi


def generate_9f236235():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    numh = unifint(diff_lb, diff_ub, (2, 14))
    numw = unifint(diff_lb, diff_ub, (2, 14))
    h = unifint(diff_lb, diff_ub, (1, 31 // numh - 1))
    w = unifint(diff_lb, diff_ub, (1, 31 // numw - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    frontcol = choice(remcols)
    remcols = remove(frontcol, cols)
    numcols = unifint(diff_lb, diff_ub, (1, min(9, numh * numw)))
    ccols = sample(remcols, numcols)
    numcells = unifint(diff_lb, diff_ub, (1, numh * numw))
    cands = asindices(canvas(-1, (numh, numw)))
    inds = asindices(canvas(-1, (h, w)))
    locs = sample(totuple(cands), numcells)
    gi = canvas(frontcol, (h * numh + numh - 1, w * numw + numw - 1))
    return gi


def generate_d8c310e9():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    p = unifint(diff_lb, diff_ub, (2, (w - 1) // 3))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    obj = set()
    for j in range(p):
        numcells = unifint(diff_lb, diff_ub, (1, h - 1))
        for ii in range(h - 1, h - numcells - 1, -1):
            loc = (ii, j)
            col = choice(ccols)
            cell = (col, loc)
            obj.add(cell)
    gi = canvas(bgc, (h, w))
    return gi


def generate_7e0986d6():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nsqcols = unifint(diff_lb, diff_ub, (1, 5))
    sqcols = sample(remcols, nsqcols)
    remcols = difference(remcols, sqcols)
    nnoisecols = unifint(diff_lb, diff_ub, (1, len(remcols)))
    noisecols = sample(remcols, nnoisecols)
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    succ = 0
    tr = 0
    maxtr = 5 * numsq
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = randint(2, 7)
        ow = randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            succ += 1
            inds = (inds - sq) - outbox(sq)
            col = choice(sqcols)
            go = fill(go, col, sq)
    gi = tuple(e for e in go)
    return gi


def generate_a64e4611():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (18, 30))
    w = unifint(diff_lb, diff_ub, (18, 30))
    bgc, noisec = sample(cols, 2)
    lb = int(0.4 * h * w)
    ub = int(0.5 * h * w)
    nbgc = unifint(diff_lb, diff_ub, (lb, ub))
    gi = canvas(noisec, (h, w))
    return gi


def generate_b782dc8a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    dlt = [('W', (-1, 0)), ('E', (1, 0)), ('S', (0, 1)), ('N', (0, -1))]
    walls = {'N': True, 'S': True, 'E': True, 'W': True}
    fullsucc = False
    while True:
        h = unifint(diff_lb, diff_ub, (3, 15))
        w = unifint(diff_lb, diff_ub, (3, 15))
        maze = [[{'x': x, 'y': y, 'walls': {**walls}} for y in range(h)] for x in range(w)]
        kk = h * w
        stck = []
        cc = maze[0][0]
        nv = 1
        while nv < kk:
            nbhs = []
            for direc, (dx, dy) in dlt:
                x2, y2 = cc['x'] + dx, cc['y'] + dy
                if 0 <= x2 < w and 0 <= y2 < h:
                    neighbour = maze[x2][y2]
                    if all(neighbour['walls'].values()):
                        nbhs.append((direc, neighbour))
            if not nbhs:
                cc = stck.pop()
                continue
            direc, next_cell = choice(nbhs)
            cc['walls'][direc] = False
            next_cell['walls'][wall_pairs[direc]] = False
            stck.append(cc)
            cc = next_cell
            nv += 1
        pathcol, wallcol, dotcol, ncol = sample(cols, 4)
        grid = [[pathcol for x in range(w * 2)]]
        for y in range(h):
            row = [pathcol]
            for x in range(w):
                row.append(wallcol)
                row.append(pathcol if maze[x][y]['walls']['E'] else wallcol)
            grid.append(row)
            row = [pathcol]
            for x in range(w):
                row.append(pathcol if maze[x][y]['walls']['S'] else wallcol)
                row.append(pathcol)
            grid.append(row)
        gi = tuple(tuple(r[1:-1]) for r in grid[1:-1])
        return gi


def generate_af902bf9():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numcols)
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    maxtr = 5 * numsq
    tr = 0
    gi = canvas(bgc, (h, w))
    return gi


def generate_a87f7484():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 30))
    num = unifint(diff_lb, diff_ub, (3, min(30 // h, 9)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, num)
    ncd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = choice((ncd, h * w - ncd))
    nc = min(max(1, nc), h * w - 1)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    origlocs = sample(totuple(inds), nc)
    canbrem = {l for l in origlocs}
    canbeadd = inds - set(origlocs)
    otherlocs = {l for l in origlocs}
    nchangesinv = unifint(diff_lb, diff_ub, (0, h * w - 1))
    nchanges = h * w - nchangesinv
    for k in range(nchanges):
        if choice((True, False)):
            if len(canbrem) > 1:
                ch = choice(totuple(canbrem))
                otherlocs = remove(ch, otherlocs)
                canbrem = remove(ch, canbrem)
            elif len(canbeadd) > 1:
                ch = choice(totuple(canbeadd))
                otherlocs = insert(ch, otherlocs)
                canbeadd = remove(ch, canbeadd)
        else:
            if len(canbeadd) > 1:
                ch = choice(totuple(canbeadd))
                otherlocs = insert(ch, otherlocs)
                canbeadd = remove(ch, canbeadd)
            elif len(canbrem) > 1:
                ch = choice(totuple(canbrem))
                otherlocs = remove(ch, otherlocs)
                canbrem = remove(ch, canbrem)
    go = fill(c, ccols[0], origlocs)
    grids = [go]
    for cc in ccols[1:]:
        grids.append(fill(c, cc, otherlocs))
    shuffle(grids)
    grids = tuple(grids)
    gi = merge(grids)
    return gi


def generate_fcc82909():
    diff_lb = 0
    diff_ub = 1
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, w // 3))
    opts = interval(0, w, 1)
    tr = 0
    maxtr = 4 * nobjs
    succ = 0
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_d9fac9be():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    bgc, noisec, ringc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_eb281b96():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 8))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    ncells = unifint(diff_lb, diff_ub, (1, h * w))
    locs = sample(totuple(inds), ncells)
    obj = {(choice(ccols), ij) for ij in locs}
    gi = paint(c, obj)
    return gi


def generate_d43fd935():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    boxh = unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccol = choice(remcols)
    remcols = remove(ccol, remcols)
    ndcols = unifint(diff_lb, diff_ub, (1, 8))
    dcols = sample(remcols, ndcols)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (h, w))
    return gi


def generate_44f52bb0():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_d22278a0():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_272f95fa():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2, 3, 4, 6))    
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, linc = sample(cols, 2)
    c = canvas(bgc, (5, 5))
    l1 = connect((1, 0), (1, 4))
    l2 = connect((3, 0), (3, 4))
    lns = l1 | l2
    gi = fill(dmirror(fill(c, linc, lns)), linc, lns)
    return gi


def generate_5c0a986e():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2))    
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    nobjs = unifint(diff_lb, diff_ub, (2, (h * w) // 10))
    gi = canvas(bgc, (h, w))
    return gi


def generate_9af7a82c():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    prods = dict()
    for a in range(1, 31, 1):
        for b in range(1, 31, 1):
            prd = a*b
            if prd in prods:
                prods[prd].append((a, b))
            else:
                prods[prd] = [(a, b)]
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    leastnc = sum(range(1, ncols + 1, 1))
    maxnc = sum(range(30, 30 - ncols, -1))
    cands = {k: v for k, v in prods.items() if leastnc <= k <= maxnc}
    options = set()
    for v in cands.values():
        for opt in v:
            options.add(opt)
    options = sorted(options, key=lambda ij: ij[0] * ij[1])
    idx = unifint(diff_lb, diff_ub, (0, len(options) - 1))
    h, w = options[idx]
    ccols = sample(cols, ncols)
    counts = list(range(1, ncols + 1, 1))
    eliginds = {ncols - 1}
    while sum(counts) < h * w:
        eligindss = sorted(eliginds, reverse=True)
        idx = unifint(diff_lb, diff_ub, (0, len(eligindss) - 1))
        idx = eligindss[idx]
        counts[idx] += 1
        if idx > 0:
            eliginds.add(idx - 1)
        if idx < ncols - 1:
            if counts[idx] == counts[idx+1] - 1:
                eliginds = eliginds - {idx}
        if counts[idx] == 30:
            eliginds = eliginds - {idx}
    gi = canvas(-1, (h, w))
    return gi


def generate_d4469b4b():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    canv = canvas(5, (3, 3))
    A = fill(canv, 0, {(1, 0), (2, 0), (1, 2), (2, 2)})
    B = fill(canv, 0, corners(asindices(canv)))
    C = fill(canv, 0, {(0, 0), (0, 1), (1, 0), (1, 1)})
    colabc = ((2, A), (1, B), (3, C))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    col, go = choice(colabc)
    gi = canvas(col, (h, w))
    return gi


def generate_bdad9b1f():
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    numh = unifint(diff_lb, diff_ub, (1, h // 2 - 1))
    numw = unifint(diff_lb, diff_ub, (1, w // 2 - 1))
    hlocs = sample(interval(2, h - 1, 1), numh)
    wlocs = sample(interval(2, w - 1, 1), numw)
    numcols = unifint(diff_lb, diff_ub, (2, 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, numcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_3345333e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = unifint(diff_lb, diff_ub, (4, h - 2))
    ow = unifint(diff_lb, diff_ub, (4, (w - 2) // 2))
    nc = unifint(diff_lb, diff_ub, (min(oh, ow), (oh * ow) // 3 * 2))
    shp = {(0, 0)}
    bounds = asindices(canvas(-1, (oh, ow)))
    for j in range(nc):
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    while height(shp) < 3 or width(shp) < 3:
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    vmshp = vmirror(shp)
    if choice((True, False)):
        vmshp = sfilter(vmshp, lambda ij: ij[1] != width(shp) - 1)
    shp = normalize(combine(shp, shift(vmshp, (0, -width(vmshp)))))
    oh, ow = shape(shp)
    bgc, objc, occcol = sample(cols, 3)
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    loc = (loci, locj)
    shp = shift(shp, loc)
    c = canvas(bgc, (h, w))
    go = fill(c, objc, shp)
    boxh = unifint(diff_lb, diff_ub, (2, oh - 1))
    boxw = unifint(diff_lb, diff_ub, (2, ow//2))
    ulci = randint(loci - 1, loci + oh - boxh + 1)
    ulcj = randint(locj + ow//2 + 1, locj + ow - boxw + 1)
    bx = backdrop(frozenset({(ulci, ulcj), (ulci + boxh - 1, ulcj + boxw - 1)}))
    gi = fill(go, occcol, bx)
    return gi


def generate_253bf280():
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (3, 30)
    colopts = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgc = choice(colopts)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    card_bounds = (0, max(1, (h * w) // 4))
    num = unifint(diff_lb, diff_ub, card_bounds)
    s = sample(inds, num)
    fgcol = choice(remove(bgc, colopts))
    gi = fill(c, fgcol, s)
    return gi


def generate_5582e5ca():
    diff_lb = 0
    diff_ub = 1
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    numc = unifint(diff_lb, diff_ub, (2, min(10, h * w - 1)))
    ccols = sample(colopts, numc)
    mostc = ccols[0]
    remcols = ccols[1:]
    leastnummostcol = (h * w) // numc + 1
    maxnummostcol = h * w - numc + 1
    nummostcold = unifint(diff_lb, diff_ub, (0, maxnummostcol - leastnummostcol))
    nummostcol = min(max(leastnummostcol, maxnummostcol - nummostcold), maxnummostcol)
    kk = len(remcols)
    remcount = h * w - nummostcol - kk
    remcounts = [1 for k in range(kk)]
    for j in range(remcount):
        cands = [idx for idx, c in enumerate(remcounts) if c < nummostcol - 1]
        if len(cands) == 0:
            break
        idx = choice(cands)
        remcounts[idx] += 1
    nummostcol = h * w - sum(remcounts)
    gi = canvas(-1, (h, w))
    return gi


def generate_a1570a43():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    oh = unifint(diff_lb, diff_ub, (3, h))
    ow = unifint(diff_lb, diff_ub, (3, w))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    crns = {(loci, locj), (loci + oh - 1, locj), (loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1)}
    cands = shift(asindices(canvas(-1, (oh-2, ow-2))), (loci+1, locj+1))
    bgc, dotc = sample(cols, 2)
    remcols = remove(bgc, remove(dotc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    gipro = canvas(bgc, (h, w))
    gipro = fill(gipro, dotc, crns)
    sp = choice(totuple(cands))
    obj = {sp}
    cands = remove(sp, cands)
    ncells = unifint(diff_lb, diff_ub, (oh + ow - 5, max(oh + ow - 5, ((oh - 2) * (ow - 2)) // 2)))
    for k in range(ncells - 1):
        obj.add(choice(totuple((cands - obj) & mapply(neighbors, obj))))
    while shape(obj) != (oh-2, ow-2):
        obj.add(choice(totuple((cands - obj) & mapply(neighbors, obj))))
    obj = {(choice(ccols), ij) for ij in obj}
    go = paint(gipro, obj)
    nperts = unifint(diff_lb, diff_ub, (1, max(h, w)))
    k = 0
    fullinds = asindices(go)
    while ulcorner(obj) == (loci+1, locj+1) or k < nperts:
        k += 1
        options = sfilter(
            neighbors((0, 0)),
            lambda ij: len(crns & shift(toindices(obj), ij)) == 0 and \
                shift(toindices(obj), ij).issubset(fullinds)
        )
        direc = choice(totuple(options))
        obj = shift(obj, direc)
    gi = paint(gipro, obj)
    return gi


def generate_f5b8619d():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (2, 15))
    ncells = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_444801d8():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numcols)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    return gi


def generate_00d62c1b():
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_10fcaaa3():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (2, 15))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 6)))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, ncols)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    locs = frozenset(sample(totuple(inds), ncells))
    obj = frozenset({(choice(ccols), ij) for ij in locs})
    gi = paint(c, obj)
    return gi


def generate_1a07d186():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nlines = unifint(diff_lb, diff_ub, (1, w // 5))
    linecols = sample(remcols, nlines)
    remcols = difference(remcols, linecols)
    nnoisecols = unifint(diff_lb, diff_ub, (0, len(remcols)))
    noisecols = sample(remcols, nnoisecols)
    locopts = interval(0, w, 1)
    locs = []
    for k in range(nlines):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        locopts = difference(locopts, interval(loc - 2, loc + 3, 1))
        locs.append(loc)
    locs = sorted(locs)
    nlines = len(locs)
    linecols = linecols[:nlines]
    gi = canvas(bgc, (h, w))
    return gi


def generate_83302e8f():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (3, 4))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh = unifint(diff_lb, diff_ub, (3, 30 // (h + 1)))
    nw = unifint(diff_lb, diff_ub, (3, 30 // (w + 1)))
    bgc, linc = sample(cols, 2)
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    return gi


def generate_98cf29f8():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    objh = unifint(diff_lb, diff_ub, (2, h - 5))
    objw = unifint(diff_lb, diff_ub, (2, w - 5))
    loci = randint(0, h - objh)
    locj = randint(0, w - objw)
    loc = (loci, locj)
    obj = backdrop(frozenset({(loci, locj), (loci + objh - 1, locj + objw - 1)}))
    bgc, objc, otherc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_1f85a75f():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = randint(3, min(8, h // 2))
    ow = randint(3, min(8, w // 2))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = randint(max(oh, ow), oh * ow)
    sp = choice(totuple(bounds))
    obj = {sp}
    cands = remove(sp, bounds)
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    bgc, objc = sample(cols, 2)
    remcols = remove(bgc, remove(objc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, ((h * w) - len(backdrop(obj))) // 4)))
    gi = canvas(bgc, (h, w))
    return gi


def generate_8eb1be9a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    oh = unifint(diff_lb, diff_ub, (2, h // 3))
    ow = unifint(diff_lb, diff_ub, (2, w))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = unifint(diff_lb, diff_ub, (2, (oh * ow) // 3 * 2))
    obj = normalize(frozenset(sample(totuple(bounds), ncells)))
    oh, ow = shape(obj)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, ncols)
    obj = frozenset({(choice(ccols), ij) for ij in obj})
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    obj = shift(obj, (loci, locj))
    c = canvas(bgc, (h, w))
    gi = paint(c, obj)
    return gi


def generate_ba26e723():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (0, 6))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    gi = canvas(0, (h, w))
    return gi


def generate_25d487eb():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 8))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_4be741c5():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    numcolors = unifint(diff_lb, diff_ub, (2, w // 3))
    ccols = sample(cols, numcolors)
    go = (tuple(ccols),)
    gi = merge(tuple(repeat(repeat(c, h), 3) for c in ccols))
    return gi


def generate_e509e548():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2, 6))
    getL = lambda h, w: connect((0, 0), (h - 1, 0)) | connect((0, 0), (0, w - 1))
    getU = lambda h, w: connect((0, 0), (0, w - 1)) | connect((0, 0), (randint(1, h - 1), 0)) | connect((0, w - 1), (randint(1, h - 1), w - 1))
    getH = lambda h, w: connect((0, 0), (0, w - 1)) | shift(connect((0, 0), (h - 1, 0)) | connect((h - 1, 0), (h - 1, randint(1, w - 1))), (0, randint(1, w - 2)))
    minshp_getter_pairs = ((2, 2, getL), (2, 3, getU), (3, 3, getH))
    colmapper = {getL: 1, getU: 6, getH: 2}
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 6))
    ccols = sample(remcols, ncols)
    nobjs = unifint(diff_lb, diff_ub, (3, (h * w) // 10))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    return gi


def generate_810b9b61():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (3,))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 6))
    ccols = sample(remcols, ncols)
    nobjs = unifint(diff_lb, diff_ub, (3, (h * w) // 10))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    return gi


def generate_6d0160f0():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (4,))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh, nw = h, w
    bgc, linc = sample(cols, 2)
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    return gi


def generate_63613498():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, sepc = sample(cols, 2)
    remcols = remove(bgc, remove(sepc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    objh = unifint(diff_lb, diff_ub, (1, h//3))
    objw = unifint(diff_lb, diff_ub, (1, w//3))
    bounds = asindices(canvas(-1, (objh, objw)))
    sp = choice(totuple(bounds))
    obj = {sp}
    ncells = unifint(diff_lb, diff_ub, (1, (objh * objw)))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    gi = canvas(bgc, (h, w))
    return gi


def generate_e5062a87():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    eligcol, objc = sample(cols, 2)
    gi = canvas(eligcol, (h, w))
    return gi


def generate_bc1d5164():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = unifint(diff_lb, diff_ub, (2, 14))
    fullh = 2 * h - 1
    fullw = 2 * w + 1
    bgc, objc = sample(cols, 2)
    inds = asindices(canvas(-1, (h, w)))
    nA = randint(1, (h - 1) * (w - 1) - 1)
    nB = randint(1, (h - 1) * (w - 1) - 1)
    nC = randint(1, (h - 1) * (w - 1) - 1)
    nD = randint(1, (h - 1) * (w - 1) - 1)
    A = sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] < w - 1)), nA)
    B = sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] > 0)), nB)
    C = sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] < w - 1)), nC)
    D = sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] > 0)), nD)
    gi = canvas(bgc, (fullh, fullw))
    return gi


def generate_11852cab():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    r1 = ((0, 0), (0, 4), (4, 0), (4, 4))
    r2 = ((2, 0), (0, 2), (4, 2), (2, 4))
    r3 = ((1, 1), (3, 1), (1, 3), (3, 3))
    r4 = ((2, 2),)
    rings = [r4, r3, r2, r1]
    bx = backdrop(frozenset(r1))
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    return gi


def generate_025d127b():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    return gi


def generate_045e512c():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (11, 30))
    w = unifint(diff_lb, diff_ub, (11, 30))
    while True:
        oh = unifint(diff_lb, diff_ub, (2, min(4, (h - 2) // 3)))
        ow = unifint(diff_lb, diff_ub, (2, min(4, (w - 2) // 3)))
        bounds = asindices(canvas(-1, (oh, ow)))
        c1 = choice(totuple(connect((0, 0), (oh - 1, 0))))
        c2 = choice(totuple(connect((0, 0), (0, ow - 1))))
        c3 = choice(totuple(connect((oh - 1, ow - 1), (oh - 1, 0))))
        c4 = choice(totuple(connect((oh - 1, ow - 1), (0, ow - 1))))
        obj = {c1, c2, c3, c4}
        remcands = totuple(bounds - obj)
        ncells = unifint(diff_lb, diff_ub, (0, len(remcands)))
        for k in range(ncells):
            loc = choice(remcands)
            obj.add(loc)
            remcands = remove(loc, remcands)
        objt = normalize(obj)
        cc = canvas(0, shape(obj))
        cc = fill(cc, 1, objt)
        if len(colorfilter(objects(cc, T, T, F), 1)) == 1:
            break
    loci = randint(oh + 1, h - 2 * oh - 1)
    locj = randint(ow + 1, w - 2 * ow - 1)
    loc = (loci, locj)
    bgc, objc = sample(cols, 2)
    remcols = remove(bgc, remove(objc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_1b60fb0c():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    odh = unifint(diff_lb, diff_ub, (2, min(h, w)//2))
    loci = randint(0, h - 2 * odh)
    locj = randint(0, w - 2 * odh)
    loc = (loci, locj)
    bgc, objc = sample(cols, 2)
    quad = canvas(bgc, (odh, odh))
    ncellsd = unifint(diff_lb, diff_ub, (0, odh ** 2 // 2))
    ncells = choice((ncellsd, odh ** 2 - ncellsd))
    ncells = min(max(1, ncells), odh ** 2 - 1)
    cells = sample(totuple(asindices(canvas(-1, (odh, odh)))), ncells)
    g1 = fill(quad, objc, cells)
    g2 = rot90(g1)
    g3 = rot90(g2)
    g4 = rot90(g3)
    c1 = shift(ofcolor(g1, objc), (0, 0))
    c2 = shift(ofcolor(g2, objc), (0, odh))
    c3 = shift(ofcolor(g3, objc), (odh, odh))
    c4 = shift(ofcolor(g4, objc), (odh, 0))
    shftamt = randint(0, odh)
    c1 = shift(c1, (0, shftamt))
    c2 = shift(c2, (shftamt, 0))
    c3 = shift(c3, (0, -shftamt))
    c4 = shift(c4, (-shftamt, 0))
    cs = (c1, c2, c3, c4)
    rempart = choice(cs)
    inobjparts = remove(rempart, cs)
    inobj = merge(set(inobjparts))
    rempart = rempart - inobj
    inobj = shift(inobj, loc)
    rempart = shift(rempart, loc)
    gi = canvas(bgc, (h, w))
    return gi


def generate_1f0c79e5():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, objc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_1f876c06():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    nlns = unifint(diff_lb, diff_ub, (1, min(min(h, w), 9)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, nlns)
    succ = 0
    tr = 0
    maxtr = 10 * nlns
    direcs = ineighbors((0, 0))
    gi = canvas(bgc, (h, w))
    return gi


def generate_22233c11():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 10))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_264363fd():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    cp = (2, 2)
    neighs = neighbors(cp)
    o1 = shift(frozenset({(0, 1), (-1, 1)}), (1, 1))
    o2 = shift(frozenset({(1, 0), (1, -1)}), (1, 1))
    o3 = shift(frozenset({(2, 1), (3, 1)}), (1, 1))
    o4 = shift(frozenset({(1, 2), (1, 3)}), (1, 1))
    mpr = {o1: (-1, 0), o2: (0, -1), o3: (1, 0), o4: (0, 1)}
    h = unifint(diff_lb, diff_ub, (15, 30))
    w = unifint(diff_lb, diff_ub, (15, 30))
    bgc, sqc, linc = sample(cols, 3)
    remcols = difference(cols, (bgc, sqc, linc))
    cpcol = choice(remcols)
    nbhcol = choice(remcols)
    nspikes = randint(1, 4)
    spikes = sample((o1, o2, o3, o4), nspikes)
    lns = merge(set(spikes))
    obj = {(cpcol, cp)} | recolor(linc, lns) | recolor(nbhcol, neighs - lns)
    loci = randint(0, h - 5)
    locj = randint(0, w - 5)
    loc = (loci, locj)
    gi = canvas(bgc, (h, w))
    return gi


def generate_29ec7d0e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    hp = unifint(diff_lb, diff_ub, (2, h//2-1))
    wp = unifint(diff_lb, diff_ub, (2, w//2-1))
    pinds = asindices(canvas(-1, (hp, wp)))
    bgc, noisec = sample(cols, 2)
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h//hp+1):
        for b in range(w//wp+1):
            loci = (a+1) + hp * a
            locj = (b+1) + wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    gi = tuple(e for e in go)
    return gi


def generate_3bd67248():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 4))
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = unifint(diff_lb, diff_ub, (3, 15))
    bgc, linc = sample(cols, 2)
    fac = unifint(diff_lb, diff_ub, (1, 30 // max(h, w)))
    gi = canvas(bgc, (h, w))
    return gi


def generate_484b58aa():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    hp = unifint(diff_lb, diff_ub, (2, h//2-1))
    wp = unifint(diff_lb, diff_ub, (2, w//2-1))
    pinds = asindices(canvas(-1, (hp, wp)))
    noisec = choice(cols)
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(-1, (h, w))
    locs = set()
    ofs = randint(1, hp - 1)
    for a in range(2*(h//hp+1)):
        for b in range(w//wp+1):
            loci = hp * a - ofs * b
            locj = wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    gi = tuple(e for e in go)
    return gi


def generate_6aa20dc0():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    od = unifint(diff_lb, diff_ub, (2, 4))
    ncellsextra = randint(1, max(1, (od ** 2 - 2) // 2))
    sinds = asindices(canvas(-1, (od, od)))
    extracells = set(sample(totuple(sinds - {(0, 0), (od - 1, od - 1)}), ncellsextra))
    extracells.add(choice(totuple(dneighbors((0, 0)) & sinds)))
    extracells.add(choice(totuple(dneighbors((od - 1, od - 1)) & sinds)))
    extracells = frozenset(extracells)
    bgc, fgc, c1, c2 = sample(cols, 4)
    obj = frozenset({(c1, (0, 0)), (c2, (od - 1, od - 1))}) | recolor(fgc, extracells)
    obj = obj | dmirror(obj)
    if choice((True, False)):
        obj = hmirror(obj)
    gi = canvas(bgc, (h, w))
    return gi


def generate_6855a6e4():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))
    bgc, objc, boxc = sample(cols, 3)
    bcanv = canvas(bgc, (h, w))
    loci = randint(0, h - fullh)
    locj = randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(sample(totuple(cands), ncells))
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    return gi


def generate_39a8645d():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (15, 30))
    w = unifint(diff_lb, diff_ub, (15, 30))
    oh = randint(2, 4)
    ow = randint(2, 4)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, oh + ow))
    ccols = sample(remcols, nobjs+1)
    mxcol = ccols[0]
    rcols = ccols[1:]
    maxnocc = unifint(diff_lb, diff_ub, (nobjs + 2, max(nobjs + 2, (h * w) // 16)))
    tr = 0
    maxtr = 10 * maxnocc
    succ = 0
    allobjs = []
    bounds = asindices(canvas(-1, (oh, ow)))
    for k in range(nobjs + 1):
        while True:
            ncells = randint(oh + ow - 1, oh * ow)
            cobj = {choice(totuple(bounds))}
            while shape(cobj) != (oh, ow) and len(cobj) < ncells:
                cobj.add(choice(totuple((bounds - cobj) & mapply(neighbors, cobj))))
            if cobj not in allobjs:
                break
        allobjs.append(frozenset(cobj))
    mcobj = normalize(allobjs[0])
    remobjs = apply(normalize, allobjs[1:])
    mxobjcounter = 0
    remobjcounter = {robj: 0 for robj in remobjs}
    gi = canvas(bgc, (h, w))
    return gi


def generate_150deff5():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 8))
    bo = {(0, 0), (0, 1), (1, 0), (1, 1)}
    ro1 = {(0, 0), (0, 1), (0, 2)}
    ro2 = {(0, 0), (1, 0), (2, 0)}
    boforb = set()
    reforb = set()
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_239be575():
    diff_lb = 0
    diff_ub = 1
    sq = {(0, 0), (1, 1), (0, 1), (1, 0)}
    cols = interval(1, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (6, 30))
        w = unifint(diff_lb, diff_ub, (6, 30))
        c = canvas(0, (h, w))
        fullcands = totuple(asindices(canvas(0, (h - 1, w - 1))))
        a = choice(fullcands)
        b = choice(remove(a, fullcands))
        mindist = unifint(diff_lb, diff_ub, (3, min(h, w) - 3))
        while not manhattan({a}, {b}) > mindist:
            a = choice(fullcands)
            b = choice(remove(a, fullcands))
        markcol, sqcol = sample(cols, 2)
        aset = shift(sq, a)
        bset = shift(sq, b)
        gi = fill(c, sqcol, aset | bset)
        return gi


def generate_0dfd9992():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    hp = unifint(diff_lb, diff_ub, (2, h//2-1))
    wp = unifint(diff_lb, diff_ub, (2, w//2-1))
    pinds = asindices(canvas(-1, (hp, wp)))
    bgc, noisec = sample(cols, 2)
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h//hp+1):
        for b in range(w//wp+1):
            loci = hp * a
            locj = wp * b
            locs.add((loci, locj))
            mf1 = identity if a % 2 == 0 else hmirror
            mf2 = identity if b % 2 == 0 else vmirror
            mf = compose(mf1, mf2)
            go = paint(go, shift(mf(pobj), (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, int((h * w) ** 0.5 // 2)))
    gi = tuple(e for e in go)
    return gi


def generate_d06dbe63():
    diff_lb = 0
    diff_ub = 1
    cols = remove(5, interval(0, 10, 1))
    obj1 = mapply(lbind(shift, frozenset({(-1, 0), (-2, 0), (-2, 1), (-2, 2)})), {(-k * 2, 2 * k) for k in range(15)})
    obj2 = mapply(lbind(shift, frozenset({(1, 0), (2, 0), (2, -1), (2, -2)})), {(2 * k, -k * 2) for k in range(15)})
    obj = obj1 | obj2
    objf = lambda ij: shift(obj, ij)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    ndots = unifint(diff_lb, diff_ub, (1, min(h, w)))
    succ = 0
    tr = 0
    maxtr = 4 * ndots
    bgc, dotc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_a3325580():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, 9))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, nobjs)
    gi = canvas(bgc, (h, w))
    return gi


def generate_1fad071e():
    diff_lb = 0
    diff_ub = 1
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nbl = randint(0, 5)
    nobjs = unifint(diff_lb, diff_ub, (nbl, max(nbl, (h * w) // 10)))
    bgc, otherc = sample(cols, 2)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    bcount = 0
    gi = canvas(bgc, (h, w))
    return gi


def generate_27a28665():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    mapping = [
    (1, {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}),
    (2, {(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)}),
    (3, {(2, 0), (0, 1), (0, 2), (1, 1), (1, 2)}),
    (6, {(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)})
    ]
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    col, obj = choice(mapping)
    bgc, objc = sample(cols, 2)
    fac = unifint(diff_lb, diff_ub, (1, min(h, w) // 3))
    go = canvas(col, (1, 1))
    gi = canvas(bgc, (h, w))
    return gi


def generate_b775ac94():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    gi = canvas(0, (1, 1))
    return gi


def generate_6f8cd79b():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_de1cd16c():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    noisec = choice(cols)
    remcols = remove(noisec, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    starterc = ccols[0]
    ccols = ccols[1:]
    gi = canvas(starterc, (h, w))
    return gi


def generate_6cf79266():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (0, 1))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    nfgcs = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(cols, nfgcs)
    gi = canvas(-1, (h, w))
    return gi


def generate_a85d4709():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 3, 4))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w3 = unifint(diff_lb, diff_ub, (1, 10))
    w = w3 * 3
    bgc, dotc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_f8a8fe49():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))
    bgc, objc, boxc = sample(cols, 3)
    bcanv = canvas(bgc, (h, w))
    loci = randint(0, h - fullh)
    locj = randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(sample(totuple(cands), ncells))
        cells = insert(choice(totuple(sfilter(cands, lambda ij: ij[0] == lowermost(cands)))), cells)
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    return gi


def generate_f8c80d96():
    diff_lb = 0
    diff_ub = 1
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    ow = randint(1, 3 if h > 10 else 2)
    oh = randint(1, 3 if w > 10 else 2)
    loci = randint(-oh+1, h-1)
    locj = randint(-ow+1, w-1)
    obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    bgc, linc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_f35d900a():
    diff_lb = 0
    diff_ub = 1
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, c1, c2 = sample(cols, 3)
    oh = unifint(diff_lb, diff_ub, (4, h))
    ow = unifint(diff_lb, diff_ub, (4, w))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = canvas(bgc, (h, w))
    return gi


def generate_ec883f72():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    ohi = unifint(diff_lb, diff_ub, (0, h - 6))
    owi = unifint(diff_lb, diff_ub, (0, w - 6))
    oh = h - 5 - ohi
    ow = w - 5 - owi
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    bgc, sqc, linc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_ea786f4a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 14))
    w = unifint(diff_lb, diff_ub, (1, 14))
    mp = (h, w)
    h = 2 * h + 1
    w = 2 * w + 1
    linc = choice(cols)
    remcols = remove(linc, cols)
    gi = canvas(linc, (h, w))
    return gi


def generate_ded97339():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, linc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_d687bc17():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, c1, c2, c3, c4 = sample(cols, 5)
    gi = canvas(bgc, (h, w))
    return gi


def generate_d90796e8():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (8, 2, 3))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc, noisec = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_a68b268e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 14))
    w = unifint(diff_lb, diff_ub, (2, 4))
    bgc, linc, c1, c2, c3, c4 = sample(cols, 6)
    canv = canvas(bgc, (h, w))
    inds = asindices(canv)
    nc1d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc1 = choice((nc1d, h * w - nc1d))
    nc1 = min(max(1, nc1), h * w - 1)
    nc2d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc2 = choice((nc2d, h * w - nc2d))
    nc2 = min(max(1, nc2), h * w - 1)
    nc3d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc3 = choice((nc3d, h * w - nc3d))
    nc3 = min(max(1, nc3), h * w - 1)
    nc4d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc4 = choice((nc4d, h * w - nc4d))
    nc4 = min(max(1, nc4), h * w - 1)
    ofc1 = sample(totuple(inds), nc1)
    ofc2 = sample(totuple(inds), nc2)
    ofc3 = sample(totuple(inds), nc3)
    ofc4 = sample(totuple(inds), nc4)
    go = fill(canv, c1, ofc1)
    go = fill(go, c2, ofc2)
    go = fill(go, c3, ofc3)
    go = fill(go, c4, ofc4)
    LR = asobject(fill(canv, c1, ofc1))
    LL = asobject(fill(canv, c2, ofc2))
    UR = asobject(fill(canv, c3, ofc3))
    UL = asobject(fill(canv, c4, ofc4))
    gi = canvas(linc, (2*h+1, 2*w+1))
    return gi


def generate_ea32f347():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2, 4))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    a = unifint(diff_lb, diff_ub, (3, 30))
    b = unifint(diff_lb, diff_ub, (2, a))
    c = unifint(diff_lb, diff_ub, (1, b))
    if c - a == 2:
        if a > 1:
            a -= 1
        elif c < min(h, w):
            c += 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_e179c5f4():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    w = unifint(diff_lb, diff_ub, (2, 10))
    h = unifint(diff_lb, diff_ub, (w+1, 30))
    bgc, linc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    return gi


def generate_aba27056():
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    bgc, sqc = sample(cols, 2)
    canv = canvas(bgc, (h, w))
    oh = randint(3, h)
    ow = unifint(diff_lb, diff_ub, (5, w - 1))
    loci = unifint(diff_lb, diff_ub, (0, h - oh))
    locj = randint(0, w - ow)
    bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    maxk = (ow - 4) // 2
    k = randint(0, maxk)
    hole = connect((loci, locj + 2 + k), (loci, locj + ow - 3 - k))
    gi = fill(canv, sqc, bx)
    return gi


def generate_e40b9e2f():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)  
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    d = unifint(diff_lb, diff_ub, (4, min(h, w) - 2))
    loci = randint(0, h - d)
    locj = randint(0, w - d)
    loc = (loci, locj)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    subg = canvas(bgc, (d, d))
    inds = asindices(subg)
    if d % 2 == 0:
        q = sfilter(inds, lambda ij: ij[0] < d//2 and ij[1] < d//2)
        cp = {(d//2-1, d//2-1), (d//2, d//2-1), (d//2-1, d//2), (d//2, d//2)}
    else:
        q = sfilter(inds, lambda ij: ij[0] < d//2 and ij[1] <= d//2)
        cp = {(d//2, d//2)} | ineighbors((d//2, d//2))
    nrings = unifint(diff_lb, diff_ub, (1, max(1, (d-2)//2)))
    rings = set()
    for k in range(nrings):
        ring = box({(k, k), (d-k-1, d-k-1)})
        rings = rings | ring
    qin = q - rings
    qout = rings & q
    ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
    tailobjcells = sample(totuple(q), ntailobjcells)
    tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
    tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    while hmirror(tailobj) == tailobj and vmirror(tailobj) == tailobj:
        ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
        tailobjcells = sample(totuple(q), ntailobjcells)
        tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
        tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    for k in range(4):
        subg = paint(subg, tailobj)
        subg = rot90(subg)
    fxobj = recolor(choice(ccols), cp)
    subg = paint(subg, fxobj)
    subgi = subg
    return gi


def generate_e8dc4411():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)  
    h = unifint(diff_lb, diff_ub, (9, 30))
    w = unifint(diff_lb, diff_ub, (9, 30))
    d = unifint(diff_lb, diff_ub, (3, min(h, w)//2-1))
    bgc, objc, remc = sample(cols, 3)
    c = canvas(bgc, (d, d))
    inds = sfilter(asindices(c), lambda ij: ij[0]>=d//2 and ij[1]>=d//2)
    ncd = unifint(diff_lb, diff_ub, (1, len(inds)//2))
    nc = choice((ncd, len(inds)-ncd))
    nc = min(max(2, nc), len(inds) - 1)
    cells = sample(totuple(inds), nc)
    cells = set(cells) | {choice(((d//2, d//2), (d//2, d//2-1)))}
    cells = cells | {(jj, ii) for ii, jj in cells}
    for k in range(4):
        c = fill(c, objc, cells)
        c = rot90(c)
    while palette(toobject(box(asindices(c)), c)) == frozenset({bgc}) and height(c) > 3:
        c = trim(c)
    obj = ofcolor(c, objc)
    od = height(obj)
    loci = randint(1, h - 2*od)
    locj = randint(1, w - 2*od)
    obj = shift(obj, (loci, locj))
    bd = backdrop(obj)
    p = 0
    while len(shift(obj, (p, p)) & bd) > 0:
        p += 1
    obj2 = shift(obj, (p, p))
    nbhs = mapply(neighbors, obj)
    while len(obj2 & nbhs) == 0:
        nbhs = mapply(neighbors, nbhs)
    indic = obj2 & nbhs
    gi = canvas(bgc, (h, w))
    return gi


def generate_ddf7fa4f():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)  
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nocc = unifint(diff_lb, diff_ub, (1, min(w // 3, (h * w) // 36)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_d07ae81c():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    lnf = lambda ij: shoot(ij, (1, 1)) | shoot(ij, (-1, -1)) | shoot(ij, (-1, 1)) | shoot(ij, (1, -1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    c1, c2, c3, c4 = sample(cols, 4)
    magiccol = 0
    gi = canvas(0, (h, w))
    return gi


def generate_b2862040():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (8,))
    while True:
        h = unifint(diff_lb, diff_ub, (10, 30))
        w = unifint(diff_lb, diff_ub, (10, 30))
        nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
        succ = 0
        tr = 0
        maxtr = 10 * nobjs
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        gi = canvas(bgc, (h, w))
        return gi


def generate_a61ba2ce():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 15))
    w = unifint(diff_lb, diff_ub, (4, 15))
    lociL = randint(2, h - 2)
    lociR = randint(2, h - 2)
    locjT = randint(2, w - 2)
    locjB = randint(2, w - 2)
    bgc, c1, c2, c3, c4 = sample(cols, 5)
    ulco = connect((0, 0), (lociL - 1, 0)) | connect((0, 0), (0, locjT - 1))
    urco = connect((0, w - 1), (0, locjT)) | connect((0, w - 1), (lociR - 1, w - 1))
    llco = connect((h - 1, 0), (lociL, 0)) | connect((h - 1, 0), (h - 1, locjB - 1))
    lrco = connect((h - 1, w - 1), (h - 1, locjB)) | connect((h - 1, w - 1), (lociR, w - 1))
    go = canvas(bgc, (h, w))
    go = fill(go, c1, ulco)
    go = fill(go, c2, urco)
    go = fill(go, c3, llco)
    go = fill(go, c4, lrco)
    fullh = unifint(diff_lb, diff_ub, (2 * h, 30))
    fullw = unifint(diff_lb, diff_ub, (2 * w, 30))
    gi = canvas(bgc, (fullh, fullw))
    return gi


def generate_bbc9ae5d():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    w = unifint(diff_lb, diff_ub, (2, 15))
    w = w * 2
    locinv = unifint(diff_lb, diff_ub, (2, w))
    locj = w - locinv
    loc = (0, locj)
    c1 = choice(cols)
    remcols = remove(c1, cols)
    ln1 = connect((0, 0), (0, locj))
    remobj = connect((0, locj+1), (0, w - 1))
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    remobj = {(choice(ccols), ij) for ij in remobj}
    gi = canvas(-1, (1, w))
    return gi


def generate_9edfc990():
    diff_lb = 0
    diff_ub = 1
    cols = interval(2, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    namt = unifint(diff_lb, diff_ub, (int(0.4 * h * w), int(0.7 * h * w)))
    gi = canvas(0, (h, w))
    return gi


def generate_a78176bb():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    nlns = unifint(diff_lb, diff_ub, (1, (h + w) // 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    succ = 0
    tr = 0
    maxtr = 10 * nlns
    gi = canvas(bgc, (h, w))
    return gi


def generate_995c5fa3():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    o1 = asindices(canvas(-1, (4, 4)))
    o2 = box(asindices(canvas(-1, (4, 4))))
    o3 = asindices(canvas(-1, (4, 4))) - {(1, 0), (2, 0), (1, 3), (2, 3)}
    o4 = o1 - shift(asindices(canvas(-1, (2, 2))), (2, 1))
    mpr = [(o1, 2), (o2, 8), (o3, 3), (o4, 4)]
    num = unifint(diff_lb, diff_ub, (1, 6))
    h = 4
    w = 4 * num + num - 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_9aec4887():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (12, 30))
    w = unifint(diff_lb, diff_ub, (12, 30))
    oh = unifint(diff_lb, diff_ub, (4, h//2-2))
    ow = unifint(diff_lb, diff_ub, (4, w//2-2))
    bgc, pc, c1, c2, c3, c4 = sample(cols, 6)
    gi = canvas(bgc, (h, w))
    return gi


def generate_846bdb03():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (12, 30))
    w = unifint(diff_lb, diff_ub, (12, 30))
    oh = unifint(diff_lb, diff_ub, (4, h//2-2))
    ow = unifint(diff_lb, diff_ub, (4, w//2-2))
    bgc, dotc, c1, c2 = sample(cols, 4)
    gi = canvas(bgc, (h, w))
    return gi


def generate_2dd70a9a():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 3))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_36fdfd69():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (4,))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 30))
    bgc, fgc, objc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_28e73c20():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (3,))
    direcmapper = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    sp = (0, w - 1)
    direc = (1, 0)
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, ncols)
    gi = canvas(-1, (h, w))
    return gi


def generate_3eda0437():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(1, 10, 1), (6,))
    h = unifint(diff_lb, diff_ub, (3, 8))
    w = unifint(diff_lb, diff_ub, (3, 30))
    if choice((True, False)):
        h, w = w, h
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    fgcs = sample(cols, ncols)
    gi = canvas(-1, (h, w))
    return gi


def generate_7447852a():
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    w = unifint(diff_lb, diff_ub, (2, 8))
    h = unifint(diff_lb, diff_ub, (w+1, 30))
    bgc, linc = sample(cols, 2)
    remcols = remove(bgc, remove(linc, cols))
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    return gi


def generate_6b9890af():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    oh = unifint(diff_lb, diff_ub, (2, 5))
    ow = unifint(diff_lb, diff_ub, (2, 5))
    h = unifint(diff_lb, diff_ub, (2*oh+2, 30))
    w = unifint(diff_lb, diff_ub, (2*ow+2, 30))
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {choice(totuple(bounds))}
    while shape(obj) != (oh, ow):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    maxfac = 1
    while oh * maxfac + 2 <= h - oh and ow * maxfac + 2 <= w - ow:
        maxfac += 1
    maxfac -= 1
    fac = unifint(diff_lb, diff_ub, (1, maxfac))
    bgc, sqc = sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    obj = {(choice(ccols), ij) for ij in obj}
    gi = canvas(bgc, (h, w))
    return gi


def generate_963e52fc():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (6, 15))
    p = unifint(diff_lb, diff_ub, (2, w // 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    obj = set()
    for j in range(p):
        ub = unifint(diff_lb, diff_ub, (0, h//2))
        ub = h//2-ub
        lb = unifint(diff_lb, diff_ub, (ub, h-1))
        numcells = unifint(diff_lb, diff_ub, (1, lb-ub+1))
        for ii in sample(interval(ub, lb+1, 1), numcells):
            loc = (ii, j)
            col = choice(ccols)
            cell = (col, loc)
            obj.add(cell)
    go = canvas(bgc, (h, w*2))
    minobj = obj | shift(obj, (0, p))
    addonw = randint(0, p)
    addon = sfilter(obj, lambda cij: cij[1][1] < addonw)
    fullobj = minobj | addon
    leftshift = randint(0, addonw)
    fullobj = shift(fullobj, (0, -leftshift))
    go = paint(go, fullobj)
    for j in range((2*w)//(2*p)+1):
        go = paint(go, shift(fullobj, (0, j * 2 * p)))
    gi = lefthalf(go)
    return gi


def generate_3e980e27():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (2, 3))
    h = unifint(diff_lb, diff_ub, (11, 30))
    w = unifint(diff_lb, diff_ub, (11, 30))
    bgc, rcol, gcol = sample(cols, 3)
    objs = []
    for (fixc, remc) in ((2, rcol), (3, gcol)):
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (2, 5))
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {choice(totuple(bounds))}
        ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(2, ncells), oh * ow)
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        fixp = choice(totuple(obj))
        rem = remove(fixp, obj)
        obj = {(fixc, fixp)} | recolor(remc, rem)
        objs.append(obj)
    robj, gobj = objs
    obj1, obj2 = sample(objs, 2)
    loci1 = randint(0, h - height(obj1) - height(obj2) - 1)
    locj1 = randint(0, w - width(obj1))
    loci2 = randint(loci1+height(obj1)+1, h - height(obj2))
    locj2 = randint(0, w - width(obj2))
    gi = canvas(bgc, (h, w))
    return gi


def generate_a8c38be5():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    goh = unifint(diff_lb, diff_ub, (9, 20))
    gow = unifint(diff_lb, diff_ub, (9, 20))
    h = unifint(diff_lb, diff_ub, (goh+4, 30))
    w = unifint(diff_lb, diff_ub, (gow+4, 30))
    bgc, sqc = sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    go = canvas(sqc, (goh, gow))
    go = fill(go, bgc, box(asindices(go)))
    loci1 = randint(2, goh-7)
    loci2 = randint(loci1+4, goh-3)
    locj1 = randint(2, gow-7)
    locj2 = randint(locj1+4, gow-3)
    f1 = hfrontier((loci1, 0))
    f2 = hfrontier((loci2, 0))
    f3 = vfrontier((0, locj1))
    f4 = vfrontier((0, locj2))
    fs = f1 | f2 | f3 | f4
    go = fill(go, sqc, fs)
    go = fill(go, bgc, {((loci1 + loci2) // 2, 1)})
    go = fill(go, bgc, {((loci1 + loci2) // 2, gow - 2)})
    go = fill(go, bgc, {(1, (locj1 + locj2) // 2)})
    go = fill(go, bgc, {(goh - 2, (locj1 + locj2) // 2)})
    objs = objects(go, T, F, T)
    objs = merge(set(recolor(choice(ccols), obj) for obj in objs))
    go = paint(go, objs)
    gi = go
    return gi


def generate_6c434453():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    nobjs = unifint(diff_lb, diff_ub, (2, (h * w) // 16))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    return gi


def generate_7837ac64():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    oh = unifint(diff_lb, diff_ub, (2, 6))
    ow = unifint(diff_lb, diff_ub, (2, 6))
    bgc, linc = sample(cols, 2)
    remcols = remove(bgc, remove(linc, cols))
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numcols)
    go = canvas(bgc, (oh, ow))
    inds = asindices(go)
    fullinds = asindices(go)
    nocc = unifint(diff_lb, diff_ub, (1, oh * ow))
    for k in range(nocc):
        mpr = {
            cc: sfilter(
                inds | mapply(neighbors, ofcolor(go, cc)),
                lambda ij: (neighbors(ij) & fullinds).issubset(inds | ofcolor(go, cc))
            ) for cc in ccols
        }
        mpr = [(kk, vv) for kk, vv in mpr.items() if len(vv) > 0]
        if len(mpr) == 0:
            break
        col, locs = choice(mpr)
        loc = choice(totuple(locs))
        go = fill(go, col, {loc})
        inds = remove(loc, inds)
    obj = fullinds - ofcolor(go, bgc)
    go = subgrid(obj, go)
    oh, ow = shape(go)
    sqsizh = unifint(diff_lb, diff_ub, (2, (30 - oh - 1) // oh))
    sqsizw = unifint(diff_lb, diff_ub, (2, (30 - ow - 1) // ow))
    fullh = oh + 1 + oh * sqsizh
    fullw = ow + 1 + ow * sqsizw
    gi = canvas(linc, (fullh, fullw))
    return gi


def generate_5ad4f10b():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    nbh = {(0, 0), (1, 0), (0, 1), (1, 1)}
    nbhs = apply(lbind(shift, nbh), {(0, 0), (-1, 0), (0, -1), (-1, -1)})
    oh = unifint(diff_lb, diff_ub, (2, 6))
    ow = unifint(diff_lb, diff_ub, (2, 6))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow - 1)
    obj = set(sample(totuple(bounds), ncells))
    while len(sfilter(obj, lambda ij: sum([len(obj & shift(nbh, ij)) < 4 for nbh in nbhs]) > 0)) == 0:
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow)
        obj = set(sample(totuple(bounds), ncells))
    obj = normalize(obj)
    oh, ow = shape(obj)
    bgc, noisec, objc = sample(cols, 3)
    go = canvas(bgc, (oh, ow))
    go = fill(go, noisec, obj)
    fac = unifint(diff_lb, diff_ub, (2, min(28//oh, 28//ow)))
    gobj = asobject(upscale(replace(go, noisec, objc), fac))
    oh, ow = shape(gobj)
    h = unifint(diff_lb, diff_ub, (oh+2, 30))
    w = unifint(diff_lb, diff_ub, (ow+2, 30))
    loci = randint(1, h - oh-1)
    locj = randint(1, w - ow-1)
    gi = canvas(bgc, (h, w))
    return gi


def generate_7df24a62():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (12, 32))
    w = unifint(diff_lb, diff_ub, (12, 32))
    oh = unifint(diff_lb, diff_ub, (3, min(7, h//3)))
    ow = unifint(diff_lb, diff_ub, (3, min(7, w//3)))
    bgc, noisec, sqc = sample(cols, 3)
    tmpg = canvas(sqc, (oh, ow))
    inbounds = backdrop(inbox(asindices(tmpg)))
    obj = {choice(totuple(inbounds))}
    while shape(obj) != (oh - 2, ow - 2):
        obj.add(choice(totuple(inbounds - obj)))
    pat = fill(tmpg, noisec, obj)
    targ = asobject(fill(canvas(bgc, (oh, ow)), noisec, obj))
    sour = asobject(pat)
    gi = canvas(bgc, (h, w))
    return gi


def generate_539a4f51():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 15))
    h, w = d, d
    gi = canvas(0, (h, w))
    return gi


def generate_ce602527():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (12, 30))
    w = unifint(diff_lb, diff_ub, (12, 30))
    bgc, c1, c2, c3 = sample(cols, 4)
    while True:
        objs = []
        for k in range(2):
            oh1 = unifint(diff_lb, diff_ub, (3, h//3-1))
            ow1 = unifint(diff_lb, diff_ub, (3, w//3-1))
            cc1 = canvas(bgc, (oh1, ow1))
            bounds1 = asindices(cc1)
            numcd1 = unifint(diff_lb, diff_ub, (0, (oh1 * ow1) // 2 - 4))
            numc1 = choice((numcd1, oh1 * ow1 - numcd1))
            numc1 = min(max(3, numc1), oh1 * ow1 - 3)
            obj1 = {choice(totuple(bounds1))}
            while len(obj1) < numc1 or shape(obj1) != (oh1, ow1):
                obj1.add(choice(totuple((bounds1 - obj1) & mapply(dneighbors, obj1))))
            objs.append(normalize(obj1))
        a, b = objs
        ag = fill(canvas(0, shape(a)), 1, a)
        bg = fill(canvas(0, shape(b)), 1, b)
        maxinh = min(height(a), height(b)) // 2 + 1
        maxinw = min(width(a), width(b)) // 2 + 1
        maxshp = (maxinh, maxinw)
        ag = crop(ag, (0, 0), maxshp)
        bg = crop(bg, (0, 0), maxshp)
        if ag != bg:
            break
    a, b = objs
    trgo = choice(objs)
    trgo2 = ofcolor(upscale(fill(canvas(0, shape(trgo)), 1, trgo), 2), 1)
    staysinh = unifint(diff_lb, diff_ub, (maxinh * 2, height(trgo) * 2))
    nout = height(trgo2) - staysinh
    loci = h - height(trgo2) + nout
    locj = randint(0, w - maxinw * 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_c8cbb738():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    gh = unifint(diff_lb, diff_ub, (3, 10))
    gw = unifint(diff_lb, diff_ub, (3, 10))
    h = unifint(diff_lb, diff_ub, (gh*2, 30))
    w = unifint(diff_lb, diff_ub, (gw*2, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_b527c5c6():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_228f6490():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nsq = unifint(diff_lb, diff_ub, (1, (h * w) // 50))
    succ = 0
    tr = 0
    maxtr = 5 * nsq
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    sqc = choice(remcols)
    remcols = remove(sqc, remcols)
    gi = canvas(bgc, (h, w))
    return gi


def generate_93b581b8():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    numocc = unifint(diff_lb, diff_ub, (1, (h * w) // 50))
    succ = 0
    tr = 0
    maxtr = 10 * numocc
    gi = canvas(bgc, (h, w))
    return gi


def generate_447fd412():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (12, 30))
    w = unifint(diff_lb, diff_ub, (12, 30))
    bgc, indic, mainc = sample(cols, 3)
    oh = unifint(diff_lb, diff_ub, (1, 4))
    ow = unifint(diff_lb, diff_ub, (1, 4))
    if oh * ow < 3:
        if choice((True, False)):
            oh = unifint(diff_lb, diff_ub, (3, 4))
        else:
            ow = unifint(diff_lb, diff_ub, (3, 4))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = unifint(diff_lb, diff_ub, (3, oh * ow))
    obj = {choice(totuple(bounds))}
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    objt = totuple(obj)
    kk = len(obj)
    nindic = randint(1, kk // 2 if kk % 2 == 1 else kk // 2 - 1)
    indicobj = set(sample(objt, nindic))
    mainobj = obj - indicobj
    obj = recolor(indic, indicobj) | recolor(mainc, mainobj)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    gi = canvas(bgc, (h, w))
    return gi


def generate_50846271():
    diff_lb = 0
    diff_ub = 1
    cols = remove(8, interval(0, 10, 1))
    cf1 = lambda d: {(d//2, 0), (d//2, d-1)} | set(sample(totuple(connect((d//2, 0), (d//2, d-1))), randint(1, d)))
    cf2 = lambda d: {(0, d//2), (d - 1, d//2)} | set(sample(totuple(connect((0, d//2), (d-1, d//2))), randint(1, d)))
    cf3 = lambda d: set(sample(totuple(remove((d//2, d//2), connect((d//2, 0), (d//2, d-1)))), randint(1, d-1))) | set(sample(totuple(remove((d//2, d//2), connect((0, d//2), (d - 1, d//2)))), randint(1, d-1)))
    cf = lambda d: choice((cf1, cf2, cf3))(d)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    dim = unifint(diff_lb, diff_ub, (1, 3))
    dim = 2 * dim + 1
    cross = connect((dim//2, 0), (dim//2, dim - 1)) | connect((0, dim//2), (dim - 1, dim//2))
    bgc, crossc, noisec = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_ae3edfdc():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2, 3, 7))
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(cols)
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    rdi = randint(1, h - 2)
    rdj = randint(1, w - 2)
    rd = (rdi, rdj)
    reminds = inds - ({rd} | neighbors(rd))
    reminds = sfilter(reminds, lambda ij: 1 <= ij[0] <= h - 2 and 1 <= ij[1] <= w - 2)
    bd = choice(totuple(reminds))
    bdi, bdj = bd
    go = fill(go, 2, {rd})
    go = fill(go, 1, {bd})
    ngd = unifint(diff_lb, diff_ub, (1, 8))
    gd = sample(totuple(neighbors(rd)), ngd)
    nod = unifint(diff_lb, diff_ub, (1, 8))
    od = sample(totuple(neighbors(bd)), nod)
    go = fill(go, 3, gd)
    go = fill(go, 7, od)
    gdmapper = {d: (3, position({rd}, {d})) for d in gd}
    odmapper = {d: (7, position({bd}, {d})) for d in od}
    mpr = {**gdmapper, **odmapper}
    ub = (len(gd) + len(od)) * ((h + w) // 5)
    ndist = unifint(diff_lb, diff_ub, (1, ub))
    gi = tuple(e for e in go)
    return gi


def generate_469497ad():
    diff_lb = 0
    diff_ub = 1
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 6))
    w = unifint(diff_lb, diff_ub, (3, 6))
    bgc, sqc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_97a05b5b():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (15, 30))
    w = unifint(diff_lb, diff_ub, (15, 30))
    sgh = randint(h//3, h//3*2)
    sgw = randint(w//3, w//3*2)
    bgc, sqc = sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    gi = canvas(bgc, (h, w))
    return gi


def generate_a5313dff():
    diff_lb = 0
    diff_ub = 1
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    return gi


def generate_780d0b14():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    nh = unifint(diff_lb, diff_ub, (2, 6))
    nw = unifint(diff_lb, diff_ub, (2, 6))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (3, 9))
    ccols = sample(remcols, ncols)
    go = canvas(-1, (nh, nw))
    obj = {(choice(ccols), ij) for ij in asindices(go)}
    go = paint(go, obj)
    while len(dedupe(go)) < nh or len(dedupe(dmirror(go))) < nw:
        obj = {(choice(ccols), ij) for ij in asindices(go)}
        go = paint(go, obj)
    h = unifint(diff_lb, diff_ub, (2*nh+nh-1, 30))
    w = unifint(diff_lb, diff_ub, (2*nw+nw-1, 30))
    hdist = [2 for k in range(nh)]
    for k in range(h - 2 * nh - nh + 1):
        idx = randint(0, nh - 1)
        hdist[idx] += 1
    wdist = [2 for k in range(nw)]
    for k in range(w - 2 * nw - nw + 1):
        idx = randint(0, nw - 1)
        wdist[idx] += 1
    gi = merge(tuple(repeat(r, c) + (repeat(bgc, nw),) for r, c in zip(go, hdist)))[:-1]
    return gi


def generate_57aa92db():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = randint(2, 5)
    ow = randint(2, 5)
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {choice(totuple(bounds))}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    fixp = choice(totuple(obj))
    bgc, fixc, mainc = sample(cols, 3)
    remcols = difference(cols, (bgc, fixc, mainc))
    gi = canvas(bgc, (h, w))
    return gi


def generate_53b68214():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (2, 6))
        w = unifint(diff_lb, diff_ub, (8, 30))
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        ncols = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(remcols, ncols)
        oh = unifint(diff_lb, diff_ub, (1, h//2))
        ow = unifint(diff_lb, diff_ub, (1, w//2-1))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (1, oh * ow))
        obj = sample(totuple(bounds), ncells)
        obj = {(choice(ccols), ij) for ij in obj}
        obj = normalize(obj)
        oh, ow = shape(obj)
        locj = randint(0, w//2)
        plcd = shift(obj, (0, locj))
        go = canvas(bgc, (10, w))
        hoffs = randint(0, ow//2 + 1)
        for k in range(10//oh+1):
            go = paint(go, shift(plcd, (k*oh, k*hoffs)))
        if len(palette(go[h:])) > 1:
            break
    gi = go[:h]
    return gi


def generate_39e1d7f9():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 10))
    w = unifint(diff_lb, diff_ub, (5, 10))
    bgc, linc, dotc = sample(cols, 3)
    remcols = difference(cols, (bgc, linc, dotc))
    gi = canvas(bgc, (h, w))
    return gi


def generate_017c7c7b():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (0, 2))
    h = unifint(diff_lb, diff_ub, (3, 10))
    w = unifint(diff_lb, diff_ub, (2, 30))
    h += h
    fgc = choice(cols)
    go = canvas(0, (h + h // 2, w))
    oh = unifint(diff_lb, diff_ub, (1, h//3*2))
    ow = unifint(diff_lb, diff_ub, (1, w))
    locj = randint(0, w - ow)
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow)
    obj = sample(totuple(bounds), ncells)
    for k in range((2*h)//oh):
        go = fill(go, 2, shift(obj, (k*oh, 0)))
    gi = replace(go[:h], 2, fgc)
    return gi


def generate_8a004b2b():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = unifint(diff_lb, diff_ub, (2, h//5))
    ow = unifint(diff_lb, diff_ub, (2, w//5))
    bounds = asindices(canvas(-1, (oh, ow)))
    bgc, cornc, ac1, ac2, objc = sample(cols, 5)
    gi = canvas(bgc, (h, w))
    return gi


def generate_49d1d64f():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 28))
    w = unifint(diff_lb, diff_ub, (2, 28))
    ncols = unifint(diff_lb, diff_ub, (1, 10))
    ccols = sample(cols, ncols)
    gi = canvas(-1, (h, w))
    return gi


def generate_890034e9():
    diff_lb = 0
    diff_ub = 1
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = randint(2, h//4)
    ow = randint(2, w//4)
    markercol = choice(cols)
    remcols = remove(markercol, cols)
    numbgc = unifint(diff_lb, diff_ub, (1, 8))
    bgcols = sample(remcols, numbgc)
    gi = canvas(0, (h, w))
    return gi


def generate_776ffc46():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, sqc, inc, outc = sample(cols, 4)
    gi = canvas(bgc, (h, w))
    return gi


def generate_e6721834():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 15))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc1, bgc2, sqc = sample(cols, 3)
    remcols = difference(cols, (bgc1, bgc2, sqc))
    gi1 = canvas(bgc1, (h, w))
    gi2 = canvas(bgc2, (h, w))
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
    tr = 0
    succ = 0
    maxtr = 5 * noccs
    gi1inds = asindices(gi1)
    gi2inds = asindices(gi2)
    go = canvas(bgc2, (h, w))
    seen = []
    while tr < maxtr and succ < noccs:
        tr += 1
        oh = randint(2, min(6, h//2))
        ow = randint(2, min(6, w//2))
        cands = sfilter(gi1inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        bounds = shift(asindices(canvas(-1, (oh, ow))), loc)
        ncells = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        obj = set(sample(totuple(bounds), ncells))
        objc = choice(remcols)
        objn = normalize(obj)
        if (objn, objc) in seen:
            continue
        seen.append(((objn, objc)))
        if bounds.issubset(gi1inds):
            succ += 1
            gi1inds = (gi1inds - bounds) - mapply(neighbors, bounds)
            gi1 = fill(gi1, sqc, bounds)
            gi1 = fill(gi1, objc, obj)
            cands2 = sfilter(gi2inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands2) == 0:
                continue
            loc2 = choice(totuple(cands2))
            bounds2 = shift(shift(bounds, invert(loc)), loc2)
            obj2 = shift(shift(obj, invert(loc)), loc2)
            if bounds2.issubset(gi2inds):
                gi2inds = (gi2inds - bounds2) - mapply(neighbors, bounds2)
                gi2 = fill(gi2, objc, obj2)
                go = fill(go, sqc, bounds2)
                go = fill(go, objc, obj2)
    gi = vconcat(gi1, gi2)
    return gi


def generate_ef135b50():
    diff_lb = 0
    diff_ub = 1
    cols = remove(9, interval(0, 10, 1))
    while True:
        h = unifint(diff_lb, diff_ub, (8, 30))
        w = unifint(diff_lb, diff_ub, (8, 30))
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        numc = unifint(diff_lb, diff_ub, (1, 8))
        ccols = sample(remcols, numc)
        gi = canvas(bgc, (h, w))
        return gi


def generate_794b24be():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2))
    mpr = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 1)}
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    nblue = randint(1, 4)
    go = canvas(bgc, (3, 3))
    for k in range(nblue):
        go = fill(go, 2, {mpr[k+1]})
    gi = canvas(bgc, (h, w))
    return gi


def generate_ff28f65a():
    diff_lb = 0
    diff_ub = 1
    cols = difference(interval(0, 10, 1), (1, 2))
    mpr = {1: (0, 0), 2: (0, 2), 3: (1, 1), 4: (2, 0), 5: (2, 2)}
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    nred = randint(1, 5)
    gi = canvas(bgc, (h, w))
    return gi


def generate_73251a56():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    while True:
        d = unifint(diff_lb, diff_ub, (10, 30))
        h, w = d, d
        noisec = choice(cols)
        remcols = remove(noisec, cols)
        nsl = unifint(diff_lb, diff_ub, (2, min(9, h//2)))
        slopes = [0] + sorted(sample(interval(1, h-1, 1), nsl - 1))
        ccols = sample(cols, nsl)
        gi = canvas(-1, (h, w))
        return gi


def generate_3631a71a():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 15))
    w = h
    bgc, patchcol = sample(cols, 2)
    patchcol = choice(cols)
    bgc = choice(remove(patchcol, cols))
    remcols = difference(cols, (bgc, patchcol))
    c = canvas(bgc, (h, w))
    inds = sfilter(asindices(c), lambda ij: ij[0] >= ij[1])
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    ncells = unifint(diff_lb, diff_ub, (1, len(inds)))
    cells = set(sample(totuple(inds), ncells))
    obj = {(choice(ccols), ij) for ij in cells}
    c = paint(dmirror(paint(c, obj)), obj)
    c = hconcat(c, vmirror(c))
    c = vconcat(c, hmirror(c))
    cutoff = 2
    go = dmirror(dmirror(c[:-cutoff])[:-cutoff])
    gi = tuple(e for e in go)
    return gi


def generate_234bbc79():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (5, 30))
        w = unifint(diff_lb, diff_ub, (6, 20))
        bgc, dotc = sample(cols, 2)
        remcols = difference(cols, (bgc, dotc))
        go = canvas(bgc, (h, 30))
        ncols = unifint(diff_lb, diff_ub, (1, 8))
        ccols = sample(remcols, ncols)
        spi = randint(0, h - 1)
        snek = [(spi, 0)]
        gi = fill(go, dotc, {(spi, 0)})
        return gi


def generate_cbded52d():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    oh = unifint(diff_lb, diff_ub, (1, 4))
    ow = unifint(diff_lb, diff_ub, (1, 4))
    numh = unifint(diff_lb, diff_ub, (3, 31 // (oh + 1)))
    numw = unifint(diff_lb, diff_ub, (3, 31 // (ow + 1)))
    bgc, linc = sample(cols, 2)
    remcols = difference(cols, (bgc, linc))
    ncols = unifint(diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
    ccols = sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = canvas(linc, (fullh, fullw))
    return gi


def generate_06df4c85():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    oh = unifint(diff_lb, diff_ub, (1, 4))
    ow = unifint(diff_lb, diff_ub, (1, 4))
    numh = unifint(diff_lb, diff_ub, (3, 31 // (oh + 1)))
    numw = unifint(diff_lb, diff_ub, (3, 31 // (ow + 1)))
    bgc, linc = sample(cols, 2)
    remcols = difference(cols, (bgc, linc))
    ncols = unifint(diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
    ccols = sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = canvas(linc, (fullh, fullw))
    return gi


def generate_90f3ed37():
    diff_lb = 0
    diff_ub = 1
    cols = remove(1, interval(0, 10, 1))
    while True:
        h = unifint(diff_lb, diff_ub, (8, 30))
        w = unifint(diff_lb, diff_ub, (8, 30))
        pathh = unifint(diff_lb, diff_ub, (1, max(1, h//4)))
        pathh = unifint(diff_lb, diff_ub, (pathh, max(1, h//4)))
        Lpatper = unifint(diff_lb, diff_ub, (1, w//7))
        Rpatper = unifint(diff_lb, diff_ub, (1, w//7))
        hh = randint(1, pathh)
        Linds = asindices(canvas(-1, (hh, Lpatper)))
        Rinds = asindices(canvas(-1, (hh, Rpatper)))
        lpatsd = unifint(diff_lb, diff_ub, (0, (hh * Lpatper) // 2))
        rpatsd = unifint(diff_lb, diff_ub, (0, (hh * Rpatper) // 2))
        lpats = choice((lpatsd, hh * Lpatper - lpatsd))
        rpats = choice((rpatsd, hh * Rpatper - rpatsd))
        lpats = min(max(Lpatper, lpats), hh * Lpatper)
        rpats = min(max(Rpatper, rpats), hh * Rpatper)
        lpat = set(sample(totuple(Linds), lpats))
        rpat = set(sample(totuple(Rinds), rpats))
        midpatw = randint(0, w-2*Lpatper-2*Rpatper)
        if midpatw == 0 or Lpatper == hh == 1:
            midpat = set()
            midpatw = 0
        else:
            midpat = set(sample(totuple(asindices(canvas(-1, (hh, midpatw)))), randint(midpatw, (hh * midpatw))))
        if shift(midpat, (0, 2*Lpatper-midpatw)).issubset(lpat):
            midpat = set()
            midpatw = 0
        loci = randint(0, h - pathh)
        lplac = shift(lpat, (loci, 0)) | shift(lpat, (loci, Lpatper))
        mplac = shift(midpat, (loci, 2*Lpatper))
        rplac = shift(rpat, (loci, 2*Lpatper+midpatw)) | shift(rpat, (loci, 2*Lpatper+midpatw+Rpatper))
        sp = 2*Lpatper+midpatw+Rpatper
        for k in range(w//Lpatper+1):
            lplac |= shift(lpat, (loci, -k*Lpatper))
        for k in range(w//Rpatper+1):
            rplac |= shift(rpat, (loci, sp+k*Rpatper))
        pat = lplac | mplac | rplac
        patn = shift(pat, (-loci, 0))
        bgc, fgc = sample(cols, 2)
        gi = canvas(bgc, (h, w))
        return gi


def generate_36d67576():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (10, 30))
        w = unifint(diff_lb, diff_ub, (10, 30))
        bgc, mainc, markerc = sample(cols, 3)
        remcols = difference(cols, (bgc, mainc, markerc))
        ncols = unifint(diff_lb, diff_ub, (1, len(remcols)))
        ccols = sample(remcols, ncols)
        gi = canvas(bgc, (h, w))
        return gi


def generate_4522001f():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 10))
    w = unifint(diff_lb, diff_ub, (3, 10))
    bgc, sqc, dotc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    return gi


def generate_72322fa7():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, 4))
    ccols = sample(remcols, 2*nobjs)
    cpairs = list(zip(ccols[:nobjs], ccols[nobjs:]))
    objs = []
    gi = canvas(bgc, (h, w))
    return gi


def generate_4290ef0e():
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)
    while True:
        d = unifint(diff_lb, diff_ub, (2, 7))
        h, w = d, d
        fullh = unifint(diff_lb, diff_ub, (4*d, 30))
        fullw = unifint(diff_lb, diff_ub, (4*d, 30))
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        ccols = sample(remcols, d)
        quad = canvas(bgc, (d+1, d+1))
        for idx, c in enumerate(ccols):
            linlen = randint(2, w-idx+1)
            quad = fill(quad, c, (connect((idx, idx), (idx+linlen-1, idx))))
            quad = fill(quad, c, (connect((idx, idx), (idx, idx+linlen-1))))
        go = canvas(bgc, (d+1, 2*d+1))
        qobj1 = asobject(quad)
        qobj2 = shift(asobject(vmirror(quad)), (0, d))
        go = paint(go, qobj1)
        go = paint(go, qobj2)
        go = vconcat(go, hmirror(go)[1:])
        if choice((True, False)):
            go = fill(go, choice(difference(remcols, ccols)), {center(asindices(go))})
        objs = partition(go)
        objs = sfilter(objs, lambda o: color(o) != bgc)
        gi = canvas(bgc, (fullh, fullw))
        return gi


def generate_6a1e5592():
    diff_lb = 0
    diff_ub = 1
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (9, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    barh = randint(3, h//3)
    maxobjh = h - barh - 1
    nobjs = unifint(diff_lb, diff_ub, (1, w//3))
    barc, bgc, objc = sample(cols, 3)
    c1 = canvas(barc, (barh, w))
    c2 = canvas(bgc, (h - barh, w))
    gi = vconcat(c1, c2)
    return gi


def generate_e73095fd():
    diff_lb = 0
    diff_ub = 1
    cols = remove(4, interval(0, 10, 1))
    while True:
        h = unifint(diff_lb, diff_ub, (10, 32))
        w = unifint(diff_lb, diff_ub, (10, 32))
        bgc, fgc = sample(cols, 2)
        gi = canvas(bgc, (h, w))
        return gi