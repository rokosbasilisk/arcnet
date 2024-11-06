from re_arc.dsl import *

def generate_e76da134() -> Grid:
    gi = canvas((1, 1, 1, 1, 1), (5, 5))
    gi = fill(gi, 2, frozenset({(1, 2), (2, 1), (3, 1), (1, 1), (2, 3), (3, 3), (3, 2), (1, 3)}))
    gi = fill(gi, 3, frozenset({(2, 2)}))
    return gi