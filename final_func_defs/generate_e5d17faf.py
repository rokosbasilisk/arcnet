from re_arc.dsl import *

def generate_grid() -> Grid:
    gi = canvas(1, (7, 7))
    gi = fill(gi, 4, frozenset({(3, 3)}))
    gi = fill(gi, 3, frozenset({(4, 4), (2, 4), (3, 4), (4, 3), (4, 2), (2, 3), (2, 2), (3, 2)}))
    gi = fill(gi, 2, frozenset({(1, 2), (2, 1), (1, 5), (3, 1), (5, 4), (5, 1), (2, 5), (4, 1), (1, 3), (3, 5), (5, 2), (5, 5), (1, 1), (1, 4), (4, 5), (5, 3)}))
    return gi