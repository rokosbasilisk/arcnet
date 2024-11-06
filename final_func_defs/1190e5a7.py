from re_arc.dsl import *
def generate_1190e5a7() -> float:
    diff_lb = 0
    diff_ub = 1
    dim_bounds = (3, 30)
    colopts = interval(0, 10, 1)
    c = canvas(bgc, (11, 28))
    nhf_bounds = (1, 11 // 3)
    nvf_bounds = (1, 28 // 3)
    hf_options = interval(1, 11 - 1, 1)
    vf_options = interval(1, 28 - 1, 1)
    hf_selection = []
    for k in range(nhf):
        hf_selection.append(hf)
        hf_options = difference(hf_options, (hf - 1, hf, hf + 1))
    vf_selection = []
    for k in range(nvf):
        vf_selection.append(vf)
        vf_options = difference(vf_options, (vf - 1, vf, vf + 1))
    remcols = remove(bgc, colopts)
    rcf = lambda x: recolor(choice(remcols), x)
    hfs = mapply(chain(rcf, hfrontier, toivec), tuple(hf_selection))
    vfs = mapply(chain(rcf, vfrontier, tojvec), tuple(vf_selection))
    gi = paint(c, combine(hfs, vfs))
    return gi