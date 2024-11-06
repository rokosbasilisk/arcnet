from re_arc.dsl import *
def generate_25d8a9c8() -> tuple:
    diff_lb = 0
    diff_ub = 1
    cols = interval(0, 10, 1)    
    gi = []
    for k in range(15):
        row = repeat(col, 23)
        if singlecol:
            gi.append(row)
        else:
            row = list(row)
            indss = interval(0, 23, 1)
            for j in sample(indss, notherc):
                row[j] = choice(remcols)
            gi.append(tuple(row))
    gi = tuple(gi)
    return gi