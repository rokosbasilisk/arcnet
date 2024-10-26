import random
import operator
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp, algorithms
from typing import Callable, Any

# ===============================
# Define Custom Types as Classes
# ===============================

class Boolean:
    def __init__(self, value: bool):
        self.value = value

    def __bool__(self):
        return self.value

    def __repr__(self):
        return f"Boolean({self.value})"


class Integer:
    def __init__(self, value: int):
        self.value = value

    def __int__(self):
        return self.value

    def __repr__(self):
        return f"Integer({self.value})"


class IntegerTuple:
    def __init__(self, value: tuple):
        self.value = value

    def __getitem__(self, index):
        return self.value[index]

    def __repr__(self):
        return f"IntegerTuple({self.value})"


class Grid:
    def __init__(self, data: tuple):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"Grid({self.data})"


class Patch:
    def __init__(self, data: frozenset):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"Patch({self.data})"


class FunctionWrapper:
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, *args):
        return self.func(*args)

    def __repr__(self):
        return f"FunctionWrapper({self.func.__name__})"


class Piece:
    def __init__(self, data: Any):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"Piece({self.data})"

# ===============================
# Define DSL Functions
# ===============================

def canvas(value: Integer, dimensions: IntegerTuple) -> Grid:
    return Grid(
        tuple(
            tuple(value.value for _ in range(dimensions.value[1]))
            for _ in range(dimensions.value[0])
        )
    )

def add_integers(a: Integer, b: Integer) -> Integer:
    return Integer(a.value + b.value)

def add_tuples(a: IntegerTuple, b: IntegerTuple) -> IntegerTuple:
    return IntegerTuple((a.value[0] + b.value[0], a.value[1] + b.value[1]))

def subtract_integers(a: Integer, b: Integer) -> Integer:
    return Integer(a.value - b.value)

def subtract_tuples(a: IntegerTuple, b: IntegerTuple) -> IntegerTuple:
    return IntegerTuple((a.value[0] - b.value[0], a.value[1] - b.value[1]))

def multiply_integers(a: Integer, b: Integer) -> Integer:
    return Integer(a.value * b.value)

def divide_integers(a: Integer, b: Integer) -> Integer:
    return Integer(a.value // b.value if b.value != 0 else 0)

def invert_integer(a: Integer) -> Integer:
    return Integer(-a.value)

def even(a: Integer) -> Boolean:
    return Boolean(a.value % 2 == 0)

def double(a: Integer) -> Integer:
    return Integer(a.value * 2)

def halve(a: Integer) -> Integer:
    return Integer(a.value // 2)

def flip_boolean(a: Boolean) -> Boolean:
    return Boolean(not a.value)

def equality_integers(a: Integer, b: Integer) -> Boolean:
    return Boolean(a.value == b.value)

def contained_integers(a: Integer, s: frozenset) -> Boolean:
    return Boolean(a.value in s)

def combine_sets(a: frozenset, b: frozenset) -> frozenset:
    return frozenset(a.union(b))

def intersection_sets(a: frozenset, b: frozenset) -> frozenset:
    return frozenset(a.intersection(b))

def difference_sets(a: frozenset, b: frozenset) -> frozenset:
    return frozenset(a.difference(b))

def dedupe_tuple(a: tuple) -> tuple:
    return tuple(dict.fromkeys(a))

def greater_integers(a: Integer, b: Integer) -> Boolean:
    return Boolean(a.value > b.value)

def size_set(a: frozenset) -> Integer:
    return Integer(len(a))

def merge_sets(a: frozenset) -> frozenset:
    return frozenset().union(*a)

def maximum_set(a: frozenset) -> Integer:
    return Integer(max(a) if a else 0)

def minimum_set(a: frozenset) -> Integer:
    return Integer(min(a) if a else 0)

def initset(a: Integer) -> frozenset:
    return frozenset({a.value})

def both_booleans(a: Boolean, b: Boolean) -> Boolean:
    return Boolean(a.value and b.value)

def either_booleans(a: Boolean, b: Boolean) -> Boolean:
    return Boolean(a.value or b.value)

def increment_integer(a: Integer) -> Integer:
    return Integer(a.value + 1)

def decrement_integer(a: Integer) -> Integer:
    return Integer(a.value - 1)

def sign_integer(a: Integer) -> Integer:
    return Integer(1 if a.value > 0 else (-1 if a.value < 0 else 0))

def positive(a: Integer) -> Boolean:
    return Boolean(a.value > 0)

def toivec(a: Integer) -> IntegerTuple:
    return IntegerTuple((a.value, 0))

def tojvec(a: Integer) -> IntegerTuple:
    return IntegerTuple((0, a.value))

def sfilter_set(a: frozenset, func: FunctionWrapper) -> frozenset:
    return frozenset(filter(func.func, a))

def mfilter_set(a: frozenset, func: FunctionWrapper) -> frozenset:
    return frozenset(map(func.func, a))

def extract_set(a: frozenset, func: FunctionWrapper) -> Integer:
    return Integer(func.func(a))

def totuple_set(a: frozenset) -> tuple:
    return tuple(a)

def first_set(a: frozenset) -> Integer:
    return Integer(next(iter(a)) if a else 0)

def last_set(a: frozenset) -> Integer:
    return Integer(max(a)) if a else Integer(0)

def insert_set(a: Integer, s: frozenset) -> frozenset:
    return frozenset(s.union({a.value}))

def remove_set(a: Integer, s: frozenset) -> frozenset:
    return frozenset(s.difference({a.value}))

def other_set(s: frozenset, a: Integer) -> Integer:
    return Integer(next(iter(s - {a.value}), 0))

def interval_integers(a: Integer, b: Integer, c: Integer) -> tuple:
    return (a.value, b.value, c.value)

def astuple(a: Integer, b: Integer) -> IntegerTuple:
    return IntegerTuple((a.value, b.value))

def pair_tuples(a: tuple, b: tuple) -> tuple:
    return (a, b)

def branch_integer(condition: Boolean, a: Integer, b: Integer) -> Integer:
    return a if condition.value else b

def compose_functions(a: FunctionWrapper, b: FunctionWrapper) -> FunctionWrapper:
    def composed(*args):
        return a.func(b.func(*args))
    return FunctionWrapper(composed)

def apply_func(a: FunctionWrapper, s: frozenset) -> frozenset:
    return frozenset(map(a.func, s))

def mostcolor(a: Grid) -> Integer:
    flat = [cell for row in a.data for cell in row]
    if not flat:
        return Integer(0)
    most_common = max(set(flat), key=flat.count)
    return Integer(most_common)

def leastcolor(a: Grid) -> Integer:
    flat = [cell for row in a.data for cell in row]
    if not flat:
        return Integer(0)
    least_common = min(set(flat), key=flat.count)
    return Integer(least_common)

def height_piece(a: Piece) -> Integer:
    return Integer(len(a.data.data))  # Assuming 'data' has 'data' attribute

def width_piece(a: Piece) -> Integer:
    return Integer(len(a.data.data[0])) if a.data.data else Integer(0)

def shape_piece(a: Piece) -> IntegerTuple:
    return IntegerTuple((len(a.data.data), len(a.data.data[0])) if a.data.data else (0, 0))

def portrait_piece(a: Piece) -> Boolean:
    return Boolean(len(a.data.data) > len(a.data.data[0]) if a.data.data else False)

def colorcount(a: Grid, color: Integer) -> Integer:
    return Integer(sum(row.count(color.value) for row in a.data))

def colorfilter(a: frozenset, color: Integer) -> frozenset:
    return frozenset(filter(lambda x: x == color.value, a))

def sizefilter(a: frozenset, size: Integer) -> frozenset:
    return frozenset(filter(lambda x: x >= size.value, a))

def asindices(a: Grid) -> frozenset:
    return frozenset((i, j) for i, row in enumerate(a.data) for j, cell in enumerate(row))

def ofcolor(a: Grid, color: Integer) -> frozenset:
    return frozenset((i, j) for i, row in enumerate(a.data) for j, cell in enumerate(row) if cell == color.value)

def ulcorner(a: frozenset) -> IntegerTuple:
    return IntegerTuple(min(a, key=lambda x: (x[0], x[1])))

def urcorner(a: frozenset) -> IntegerTuple:
    return IntegerTuple(min(a, key=lambda x: (x[0], -x[1])))

def llcorner(a: frozenset) -> IntegerTuple:
    return IntegerTuple(max(a, key=lambda x: (x[0], x[1])))

def lrcorner(a: frozenset) -> IntegerTuple:
    return IntegerTuple(max(a, key=lambda x: (x[0], -x[1])))

def toindices(a: Patch) -> frozenset:
    return a.data

def recolor(color: Integer, patch: Patch) -> Patch:
    # Placeholder implementation
    return Patch(patch.data)

def shift(patch: Patch, offset: IntegerTuple) -> Patch:
    return Patch(frozenset((i + offset.value[0], j + offset.value[1]) for i, j in patch.data))

def normalize(patch: Patch) -> Patch:
    min_i = min(i for i, j in patch.data) if patch.data else 0
    min_j = min(j for i, j in patch.data) if patch.data else 0
    return Patch(frozenset((i - min_i, j - min_j) for i, j in patch.data))

def dneighbors(a: IntegerTuple) -> frozenset:
    i, j = a.value
    return frozenset([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])

def ineighbors(a: IntegerTuple) -> frozenset:
    i, j = a.value
    return frozenset([(i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)])

def neighbors(a: IntegerTuple) -> frozenset:
    return dneighbors(a).union(ineighbors(a))

def objects(a: Grid, fg: Boolean, bg: Boolean, others: Boolean) -> frozenset:
    # Placeholder implementation
    return frozenset()

def partition(a: Grid) -> frozenset:
    # Placeholder implementation
    return frozenset()

def fgpartition(a: Grid) -> frozenset:
    # Placeholder implementation
    return frozenset()

def uppermost(a: frozenset) -> Integer:
    return Integer(min(i for i, j in a) if a else 0)

def lowermost(a: frozenset) -> Integer:
    return Integer(max(i for i, j in a) if a else 0)

def leftmost(a: frozenset) -> Integer:
    return Integer(min(j for i, j in a) if a else 0)

def rightmost(a: frozenset) -> Integer:
    return Integer(max(j for i, j in a) if a else 0)

def square_piece(a: Piece) -> Boolean:
    shape = shape_piece(a)
    return Boolean(shape.value[0] == shape.value[1])

def vline(a: frozenset) -> Boolean:
    cols = {j for i, j in a}
    return Boolean(len(cols) == 1)

def hline(a: frozenset) -> Boolean:
    rows = {i for i, j in a}
    return Boolean(len(rows) == 1)

def hmatching(a: frozenset, b: frozenset) -> Boolean:
    return Boolean(False)  # Placeholder

def vmatching(a: frozenset, b: frozenset) -> Boolean:
    return Boolean(False)  # Placeholder

def manhattan(a: frozenset, b: frozenset) -> Integer:
    return Integer(0)  # Placeholder

def adjacent(a: frozenset, b: frozenset) -> Boolean:
    return Boolean(False)  # Placeholder

def bordering(a: frozenset, grid: Grid) -> Boolean:
    return Boolean(False)  # Placeholder

def centerofmass(a: frozenset) -> IntegerTuple:
    if not a:
        return IntegerTuple((0, 0))
    avg_i = sum(i for i, j in a) / len(a)
    avg_j = sum(j for i, j in a) / len(a)
    return IntegerTuple((int(avg_i), int(avg_j)))

def palette(a: Grid) -> frozenset:
    return frozenset(set(cell for row in a.data for cell in row))

def numcolors(a: Grid) -> Integer:
    return Integer(len(set(cell for row in a.data for cell in row)))

def color(a: frozenset) -> Integer:
    return Integer(0)  # Placeholder

def toobject(a: frozenset, grid: Grid) -> frozenset:
    return frozenset()

def asobject(a: Grid) -> frozenset:
    return frozenset()

def hmirror_piece(a: Piece) -> Piece:
    return Piece(a.data)

def vmirror_piece(a: Piece) -> Piece:
    return Piece(a.data)

def dmirror_piece(a: Piece) -> Piece:
    return Piece(a.data)

def cmirror_piece(a: Piece) -> Piece:
    return Piece(a.data)

def paint(a: Grid, s: frozenset) -> Grid:
    return Grid(a.data)

def underfill(a: Grid, color: Integer, patch: Patch) -> Grid:
    return Grid(a.data)

def underpaint(a: Grid, s: frozenset) -> Grid:
    return Grid(a.data)

def hupscale(a: Grid, factor: Integer) -> Grid:
    return Grid(a.data)

def vupscale(a: Grid, factor: Integer) -> Grid:
    return Grid(a.data)

def upscale_grid(a: Grid, factor: Integer) -> Grid:
    return Grid(a.data)

def upscale_patch(a: Patch, factor: Integer) -> Patch:
    return Patch(a.data)

def downscale(a: Grid, factor: Integer) -> Grid:
    return Grid(a.data)

def hconcat(a: Grid, b: Grid) -> Grid:
    return Grid(a.data)

def vconcat(a: Grid, b: Grid) -> Grid:
    return Grid(a.data)

def subgrid(a: Patch, grid: Grid) -> Grid:
    return Grid(grid.data)

def hsplit(a: Grid, index: Integer) -> tuple:
    return (a, a)

def vsplit(a: Grid, index: Integer) -> tuple:
    return (a, a)

def cellwise(a: Grid, b: Grid, op: Integer) -> Grid:
    return Grid(a.data)

def switch(a: Grid, from_color: Integer, to_color: Integer) -> Grid:
    return Grid(a.data)

def center(a: frozenset) -> IntegerTuple:
    return IntegerTuple((0, 0))

def position(a: frozenset, b: frozenset) -> IntegerTuple:
    return IntegerTuple((0, 0))

def index_grid(a: Grid, pos: IntegerTuple) -> Integer:
    return Integer(a.data[pos.value[0]][pos.value[1]] if 0 <= pos.value[0] < len(a.data) and 0 <= pos.value[1] < len(a.data[0]) else 0)

def corners(a: frozenset) -> frozenset:
    return frozenset()

def connect(a: IntegerTuple, b: IntegerTuple) -> frozenset:
    ai, aj = a.value
    bi, bj = b.value
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset({(ai, j) for j in range(sj, ej)})
    elif aj == bj:
        return frozenset({(i, aj) for i in range(si, ei)})
    elif bi - ai == bj - aj:
        return frozenset({(i, j) for i, j in zip(range(si, ei), range(sj, ej))})
    elif bi - ai == aj - bj:
        return frozenset({(i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1))})
    return frozenset()

def cover(a: Grid, s: frozenset) -> Grid:
    return Grid(a.data)

def trim(a: Grid) -> Grid:
    return Grid(a.data)

def move(a: Grid, s: frozenset, offset: IntegerTuple) -> Grid:
    return Grid(a.data)

def tophalf(a: Grid) -> Grid:
    return Grid(a.data[:len(a.data)//2])

def bottomhalf(a: Grid) -> Grid:
    return Grid(a.data[len(a.data)//2:])

def lefthalf(a: Grid) -> Grid:
    return Grid(a.data)

def righthalf(a: Grid) -> Grid:
    return Grid(a.data)

def vfrontier(a: IntegerTuple) -> frozenset:
    return frozenset()

def hfrontier(a: IntegerTuple) -> frozenset:
    return frozenset()

def backdrop(a: frozenset) -> frozenset:
    return frozenset()

def delta(a: frozenset) -> frozenset:
    return frozenset()

def gravitate(a: frozenset, b: frozenset) -> IntegerTuple:
    return IntegerTuple((0, 0))

def inbox(a: frozenset) -> frozenset:
    return frozenset()

def outbox(a: frozenset) -> frozenset:
    return frozenset()

def box(a: frozenset) -> frozenset:
    return frozenset()

def shoot(a: IntegerTuple, b: IntegerTuple) -> frozenset:
    return frozenset()

def occurrences(a: Grid, b: frozenset) -> frozenset:
    return frozenset()

def frontiers(a: Grid) -> frozenset:
    return frozenset()

def vperiod(a: frozenset) -> Integer:
    return Integer(0)

def compress(a: Grid) -> Grid:
    return Grid(a.data)

def hperiod(a: frozenset) -> Integer:
    return Integer(0)

# ===============================
# Define Fitness and Individual
# ===============================

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# ===============================
# Define the Primitive Set
# ===============================

pset = gp.PrimitiveSetTyped("MAIN", [], Grid)

# ===============================
# Add Primitives from DSL
# ===============================

pset.addPrimitive(canvas, [Integer, IntegerTuple], Grid, name="canvas")
pset.addPrimitive(fill, [Grid, Integer, Patch], Grid, name="fill")
pset.addPrimitive(replace, [Grid, Integer, Integer], Grid, name="replace")
pset.addPrimitive(rot90, [Grid], Grid, name="rot90")
pset.addPrimitive(rot180, [Grid], Grid, name="rot180")
pset.addPrimitive(rot270, [Grid], Grid, name="rot270")
pset.addPrimitive(crop, [Grid, IntegerTuple, IntegerTuple], Grid, name="crop")
pset.addPrimitive(add_integers, [Integer, Integer], Integer, name="add_integers")
pset.addPrimitive(add_tuples, [IntegerTuple, IntegerTuple], IntegerTuple, name="add_tuples")
pset.addPrimitive(subtract_integers, [Integer, Integer], Integer, name="subtract_integers")
pset.addPrimitive(subtract_tuples, [IntegerTuple, IntegerTuple], IntegerTuple, name="subtract_tuples")
pset.addPrimitive(multiply_integers, [Integer, Integer], Integer, name="multiply_integers")
pset.addPrimitive(divide_integers, [Integer, Integer], Integer, name="divide_integers")
pset.addPrimitive(invert_integer, [Integer], Integer, name="invert_integer")
pset.addPrimitive(even, [Integer], Boolean, name="even")
pset.addPrimitive(double, [Integer], Integer, name="double")
pset.addPrimitive(halve, [Integer], Integer, name="halve")
pset.addPrimitive(flip_boolean, [Boolean], Boolean, name="flip_boolean")
pset.addPrimitive(equality_integers, [Integer, Integer], Boolean, name="equality_integers")
pset.addPrimitive(contained_integers, [Integer, frozenset], Boolean, name="contained_integers")
pset.addPrimitive(combine_sets, [frozenset, frozenset], frozenset, name="combine_sets")
pset.addPrimitive(intersection_sets, [frozenset, frozenset], frozenset, name="intersection_sets")
pset.addPrimitive(difference_sets, [frozenset, frozenset], frozenset, name="difference_sets")
pset.addPrimitive(dedupe_tuple, [tuple], tuple, name="dedupe_tuple")
pset.addPrimitive(greater_integers, [Integer, Integer], Boolean, name="greater_integers")
pset.addPrimitive(size_set, [frozenset], Integer, name="size_set")
pset.addPrimitive(merge_sets, [frozenset], frozenset, name="merge_sets")
pset.addPrimitive(maximum_set, [frozenset], Integer, name="maximum_set")
pset.addPrimitive(minimum_set, [frozenset], Integer, name="minimum_set")
pset.addPrimitive(initset, [Integer], frozenset, name="initset")
pset.addPrimitive(both_booleans, [Boolean, Boolean], Boolean, name="both_booleans")
pset.addPrimitive(either_booleans, [Boolean, Boolean], Boolean, name="either_booleans")
pset.addPrimitive(increment_integer, [Integer], Integer, name="increment_integer")
pset.addPrimitive(decrement_integer, [Integer], Integer, name="decrement_integer")
pset.addPrimitive(sign_integer, [Integer], Integer, name="sign_integer")
pset.addPrimitive(positive, [Integer], Boolean, name="positive")
pset.addPrimitive(toivec, [Integer], IntegerTuple, name="toivec")
pset.addPrimitive(tojvec, [Integer], IntegerTuple, name="tojvec")
pset.addPrimitive(sfilter_set, [frozenset, FunctionWrapper], frozenset, name="sfilter_set")
pset.addPrimitive(mfilter_set, [frozenset, FunctionWrapper], frozenset, name="mfilter_set")
pset.addPrimitive(extract_set, [frozenset, FunctionWrapper], Integer, name="extract_set")
pset.addPrimitive(totuple_set, [frozenset], tuple, name="totuple_set")
pset.addPrimitive(first_set, [frozenset], Integer, name="first_set")
pset.addPrimitive(last_set, [frozenset], Integer, name="last_set")
pset.addPrimitive(insert_set, [Integer, frozenset], frozenset, name="insert_set")
pset.addPrimitive(remove_set, [Integer, frozenset], frozenset, name="remove_set")
pset.addPrimitive(other_set, [frozenset, Integer], Integer, name="other_set")
pset.addPrimitive(interval_integers, [Integer, Integer, Integer], tuple, name="interval_integers")
pset.addPrimitive(astuple, [Integer, Integer], IntegerTuple, name="astuple")
pset.addPrimitive(pair_tuples, [tuple, tuple], tuple, name="pair_tuples")
pset.addPrimitive(branch_integer, [Boolean, Integer, Integer], Integer, name="branch_integer")
pset.addPrimitive(compose_functions, [FunctionWrapper, FunctionWrapper], FunctionWrapper, name="compose_functions")
pset.addPrimitive(apply_func, [FunctionWrapper, frozenset], frozenset, name="apply_func")
pset.addPrimitive(mostcolor, [Grid], Integer, name="mostcolor")
pset.addPrimitive(leastcolor, [Grid], Integer, name="leastcolor")
pset.addPrimitive(height_piece, [Piece], Integer, name="height_piece")
pset.addPrimitive(width_piece, [Piece], Integer, name="width_piece")
pset.addPrimitive(shape_piece, [Piece], IntegerTuple, name="shape_piece")
pset.addPrimitive(portrait_piece, [Piece], Boolean, name="portrait_piece")
pset.addPrimitive(colorcount, [Grid, Integer], Integer, name="colorcount")
pset.addPrimitive(colorfilter, [frozenset, Integer], frozenset, name="colorfilter")
pset.addPrimitive(sizefilter, [frozenset, Integer], frozenset, name="sizefilter")
pset.addPrimitive(asindices, [Grid], frozenset, name="asindices")
pset.addPrimitive(ofcolor, [Grid, Integer], frozenset, name="ofcolor")
pset.addPrimitive(ulcorner, [frozenset], IntegerTuple, name="ulcorner")
pset.addPrimitive(urcorner, [frozenset], IntegerTuple, name="urcorner")
pset.addPrimitive(llcorner, [frozenset], IntegerTuple, name="llcorner")
pset.addPrimitive(lrcorner, [frozenset], IntegerTuple, name="lrcorner")
pset.addPrimitive(toindices, [Patch], frozenset, name="toindices")
pset.addPrimitive(recolor, [Integer, Patch], Patch, name="recolor")
pset.addPrimitive(shift, [Patch, IntegerTuple], Patch, name="shift")
pset.addPrimitive(normalize, [Patch], Patch, name="normalize")
pset.addPrimitive(dneighbors, [IntegerTuple], frozenset, name="dneighbors")
pset.addPrimitive(ineighbors, [IntegerTuple], frozenset, name="ineighbors")
pset.addPrimitive(neighbors, [IntegerTuple], frozenset, name="neighbors")
pset.addPrimitive(objects, [Grid, Boolean, Boolean, Boolean], frozenset, name="objects")
pset.addPrimitive(partition, [Grid], frozenset, name="partition")
pset.addPrimitive(fgpartition, [Grid], frozenset, name="fgpartition")
pset.addPrimitive(uppermost, [frozenset], Integer, name="uppermost")
pset.addPrimitive(lowermost, [frozenset], Integer, name="lowermost")
pset.addPrimitive(leftmost, [frozenset], Integer, name="leftmost")
pset.addPrimitive(rightmost, [frozenset], Integer, name="rightmost")
pset.addPrimitive(square_piece, [Piece], Boolean, name="square_piece")
pset.addPrimitive(vline, [frozenset], Boolean, name="vline")
pset.addPrimitive(hline, [frozenset], Boolean, name="hline")
pset.addPrimitive(hmatching, [frozenset, frozenset], Boolean, name="hmatching")
pset.addPrimitive(vmatching, [frozenset, frozenset], Boolean, name="vmatching")
pset.addPrimitive(manhattan, [frozenset, frozenset], Integer, name="manhattan")
pset.addPrimitive(adjacent, [frozenset, frozenset], Boolean, name="adjacent")
pset.addPrimitive(bordering, [frozenset, Grid], Boolean, name="bordering")
pset.addPrimitive(centerofmass, [frozenset], IntegerTuple, name="centerofmass")
pset.addPrimitive(palette, [Grid], frozenset, name="palette")
pset.addPrimitive(numcolors, [Grid], Integer, name="numcolors")
pset.addPrimitive(color, [frozenset], Integer, name="color")
pset.addPrimitive(toobject, [frozenset, Grid], frozenset, name="toobject")
pset.addPrimitive(asobject, [Grid], frozenset, name="asobject")
pset.addPrimitive(hmirror_piece, [Piece], Piece, name="hmirror_piece")
pset.addPrimitive(vmirror_piece, [Piece], Piece, name="vmirror_piece")
pset.addPrimitive(dmirror_piece, [Piece], Piece, name="dmirror_piece")
pset.addPrimitive(cmirror_piece, [Piece], Piece, name="cmirror_piece")
pset.addPrimitive(paint, [Grid, frozenset], Grid, name="paint")
pset.addPrimitive(underfill, [Grid, Integer, Patch], Grid, name="underfill")
pset.addPrimitive(underpaint, [Grid, frozenset], Grid, name="underpaint")
pset.addPrimitive(hupscale, [Grid, Integer], Grid, name="hupscale")
pset.addPrimitive(vupscale, [Grid, Integer], Grid, name="vupscale")
pset.addPrimitive(upscale_grid, [Grid, Integer], Grid, name="upscale_grid")
pset.addPrimitive(upscale_patch, [Patch, Integer], Patch, name="upscale_patch")
pset.addPrimitive(downscale, [Grid, Integer], Grid, name="downscale")
pset.addPrimitive(hconcat, [Grid, Grid], Grid, name="hconcat")
pset.addPrimitive(vconcat, [Grid, Grid], Grid, name="vconcat")
pset.addPrimitive(subgrid, [Patch, Grid], Grid, name="subgrid")
pset.addPrimitive(hsplit, [Grid, Integer], tuple, name="hsplit")
pset.addPrimitive(vsplit, [Grid, Integer], tuple, name="vsplit")
pset.addPrimitive(cellwise, [Grid, Grid, Integer], Grid, name="cellwise")
pset.addPrimitive(switch, [Grid, Integer, Integer], Grid, name="switch")
pset.addPrimitive(center, [frozenset], IntegerTuple, name="center")
pset.addPrimitive(position, [frozenset, frozenset], IntegerTuple, name="position")
pset.addPrimitive(index_grid, [Grid, IntegerTuple], Integer, name="index_grid")
pset.addPrimitive(corners, [frozenset], frozenset, name="corners")
pset.addPrimitive(connect, [IntegerTuple, IntegerTuple], frozenset, name="connect")
pset.addPrimitive(cover, [Grid, frozenset], Grid, name="cover")
pset.addPrimitive(trim, [Grid], Grid, name="trim")
pset.addPrimitive(move, [Grid, frozenset, IntegerTuple], Grid, name="move")
pset.addPrimitive(tophalf, [Grid], Grid, name="tophalf")
pset.addPrimitive(bottomhalf, [Grid], Grid, name="bottomhalf")
pset.addPrimitive(lefthalf, [Grid], Grid, name="lefthalf")
pset.addPrimitive(righthalf, [Grid], Grid, name="righthalf")
pset.addPrimitive(vfrontier, [IntegerTuple], frozenset, name="vfrontier")
pset.addPrimitive(hfrontier, [IntegerTuple], frozenset, name="hfrontier")
pset.addPrimitive(backdrop, [frozenset], frozenset, name="backdrop")
pset.addPrimitive(delta, [frozenset], frozenset, name="delta")
pset.addPrimitive(gravitate, [frozenset, frozenset], IntegerTuple, name="gravitate")
pset.addPrimitive(inbox, [frozenset], frozenset, name="inbox")
pset.addPrimitive(outbox, [frozenset], frozenset, name="outbox")
pset.addPrimitive(box, [frozenset], frozenset, name="box")
pset.addPrimitive(shoot, [IntegerTuple, IntegerTuple], frozenset, name="shoot")
pset.addPrimitive(occurrences, [Grid, frozenset], frozenset, name="occurrences")
pset.addPrimitive(frontiers, [Grid], frozenset, name="frontiers")
pset.addPrimitive(vperiod, [frozenset], Integer, name="vperiod")
pset.addPrimitive(compress, [Grid], Grid, name="compress")
pset.addPrimitive(hperiod, [frozenset], Integer, name="hperiod")
pset.addPrimitive(compose_functions, [FunctionWrapper, FunctionWrapper], FunctionWrapper, name="compose_functions")

# ===============================
# Add Terminals (Constants)
# ===============================

for color in range(10):
    pset.addTerminal(Integer(color), Integer, name=f"color{color}")

pset.addTerminal(IntegerTuple((5, 5)), IntegerTuple, name="dims_5_5")
pset.addTerminal(IntegerTuple((0, 0)), IntegerTuple, name="start_0_0")
pset.addTerminal(IntegerTuple((1, 1)), IntegerTuple, name="unity")
pset.addTerminal(IntegerTuple((1, 0)), IntegerTuple, name="down")
pset.addTerminal(IntegerTuple((0, 1)), IntegerTuple, name="right")
pset.addTerminal(IntegerTuple((-1, 0)), IntegerTuple, name="up")
pset.addTerminal(IntegerTuple((0, -1)), IntegerTuple, name="left")
pset.addTerminal(Boolean(True), Boolean, name="true")
pset.addTerminal(Boolean(False), Boolean, name="false")

# ===============================
# Set Up the Toolbox
# ===============================

toolbox = base.Toolbox()
toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# ===============================
# Fitness Function
# ===============================

def compute_fitness(generated_grid: Grid, target_grid: Grid) -> float:
    try:
        if len(generated_grid.data) != len(target_grid.data):
            return float('inf')
        mse = 0
        for row_gen, row_tar in zip(generated_grid.data, target_grid.data):
            if len(row_gen) != len(row_tar):
                return float('inf')
            for cell_gen, cell_tar in zip(row_gen, row_tar):
                mse += (cell_gen - cell_tar) ** 2
        return mse / (len(target_grid.data) * len(target_grid.data[0]))
    except Exception as e:
        print(f"Error in compute_fitness: {e}")
        return float('inf')

# ===============================
# Evaluation Function
# ===============================

TARGET_GRID = Grid((
    (0, 1, 1, 0, 0),
    (1, 2, 2, 1, 0),
    (1, 2, 2, 1, 0),
    (0, 1, 1, 0, 0),
    (0, 0, 0, 0, 0),
))

def eval_individual(individual: creator.Individual) -> tuple:
    func: Callable[[], Grid] = toolbox.compile(expr=individual)
    try:
        generated_grid = func()
        fitness = compute_fitness(generated_grid, TARGET_GRID)
    except Exception as e:
        print(f"Exception during evaluation of individual {individual}: {e}")
        fitness = float('inf')
    return (fitness,)

toolbox.register("evaluate", eval_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# ===============================
# Genetic Programming Parameters
# ===============================

POPULATION_SIZE = 300
GENERATIONS = 40
CX_PROB = 0.5    # Crossover probability
MUT_PROB = 0.2   # Mutation probability

# ===============================
# Visualization Function
# ===============================

def visualize_grid(grid: Grid, title: str = "Grid", fig_size: tuple = (5, 5)):
    h, w = len(grid.data), len(grid.data[0])
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(grid.data, cmap='tab10', vmin=0, vmax=9)
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.show()

# ===============================
# Main Evolutionary Loop
# ===============================

def main():
    random.seed(42)
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    mstats.register("std", np.std)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB, ngen=GENERATIONS, stats=mstats, halloffame=hof, verbose=True)
    print("\nBest individual:")
    print(hof[0])
    func: Callable[[], Grid] = toolbox.compile(expr=hof[0])
    try:
        generated_grid = func()
        print("\nGenerated Grid:")
        for row in generated_grid.data:
            print(row)
        visualize_grid(generated_grid, title="Best Generated Grid")
        visualize_grid(TARGET_GRID, title="Target Grid")
    except Exception as e:
        print(f"Error executing the best individual: {e}")

# ===============================
# Execute the Program
# ===============================

if __name__ == "__main__":
    main()

