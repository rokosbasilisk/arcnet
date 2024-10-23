from constants import *


def crop(grid: Grid, start: IntegerTuple, dims: IntegerTuple) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple((r[start[1]:start[1] + dims[1]] for r in grid[start[0]:start[0] + dims[0]]))

def rot90(grid: Grid) -> Grid:
    """ quarter clockwise rotation """
    return tuple((row for row in zip(*grid[::-1])))

def rot180(grid: Grid) -> Grid:
    """ half rotation """
    return tuple((tuple(row[::-1]) for row in grid[::-1]))

def rot270(grid: Grid) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple((tuple(row[::-1]) for row in zip(*grid[::-1])))[::-1]

def fill(grid: Grid, value: Integer, patch: Patch) -> Grid:
    """ fill value at indices """
    h, w = (len(grid), len(grid[0]))
    grid_filled = list((list(row) for row in grid))
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return tuple((tuple(row) for row in grid_filled))

def paint(grid: Grid, obj: Object) -> Grid:
    """ paint object to grid """
    h, w = (len(grid), len(grid[0]))
    grid_painted = list((list(row) for row in grid))
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            grid_painted[i][j] = value
    return tuple((tuple(row) for row in grid_painted))

def underfill(grid: Grid, value: Integer, patch: Patch) -> Grid:
    """ fill value at indices that are background """
    h, w = (len(grid), len(grid[0]))
    bg = mostcolor(grid)
    grid_filled = list((list(row) for row in grid))
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            if grid_filled[i][j] == bg:
                grid_filled[i][j] = value
    return tuple((tuple(row) for row in grid_filled))

def underpaint(grid: Grid, obj: Object) -> Grid:
    """ paint object to grid where there is background """
    h, w = (len(grid), len(grid[0]))
    bg = mostcolor(grid)
    grid_painted = list((list(row) for row in grid))
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            if grid_painted[i][j] == bg:
                grid_painted[i][j] = value
    return tuple((tuple(row) for row in grid_painted))

def hupscale(grid: Grid, factor: Integer) -> Grid:
    """ upscale grid horizontally """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_row = tuple()
        for value in row:
            upscaled_row = upscaled_row + tuple((value for num in range(factor)))
        upscaled_grid = upscaled_grid + (upscaled_row,)
    return upscaled_grid

def vupscale(grid: Grid, factor: Integer) -> Grid:
    """ upscale grid vertically """
    upscaled_grid = tuple()
    for row in grid:
        upscaled_grid = upscaled_grid + tuple((row for num in range(factor)))
    return upscaled_grid

def downscale(grid: Grid, factor: Integer) -> Grid:
    """ downscale grid """
    h, w = (len(grid), len(grid[0]))
    downscaled_grid = tuple()
    for i in range(h):
        downscaled_row = tuple()
        for j in range(w):
            if j % factor == 0:
                downscaled_row = downscaled_row + (grid[i][j],)
        downscaled_grid = downscaled_grid + (downscaled_row,)
    h = len(downscaled_grid)
    downscaled_grid2 = tuple()
    for i in range(h):
        if i % factor == 0:
            downscaled_grid2 = downscaled_grid2 + (downscaled_grid[i],)
    return downscaled_grid2

def hconcat(a: Grid, b: Grid) -> Grid:
    """ concatenate two grids horizontally """
    return tuple((i + j for i, j in zip(a, b)))

def vconcat(a: Grid, b: Grid) -> Grid:
    """ concatenate two grids vertically """
    return a + b

def subgrid(patch: Patch, grid: Grid) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))

def cellwise(a: Grid, b: Grid, fallback: Integer) -> Grid:
    """ cellwise match of two grids """
    h, w = (len(a), len(a[0]))
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            value = a_value if a_value == b[i][j] else fallback
            row = row + (value,)
        resulting_grid = resulting_grid + (row,)
    return resulting_grid

def replace(grid: Grid, replacee: Integer, replacer: Integer) -> Grid:
    """ color substitution """
    return tuple((tuple((replacer if v == replacee else v for v in r)) for r in grid))

def switch(grid: Grid, a: Integer, b: Integer) -> Grid:
    """ color switching """
    return tuple((tuple((v if v != a and v != b else {a: b, b: a}[v] for v in r)) for r in grid))

def canvas(value: Integer, dimensions: IntegerTuple) -> Grid:
    """ grid construction """
    return tuple((tuple((value for j in range(dimensions[1]))) for i in range(dimensions[0])))

def cover(grid: Grid, patch: Patch) -> Grid:
    """ remove object from grid """
    return fill(grid, mostcolor(grid), toindices(patch))

def trim(grid: Grid) -> Grid:
    """ trim border of grid """
    return tuple((r[1:-1] for r in grid[1:-1]))

def move(grid: Grid, obj: Object, offset: IntegerTuple) -> Grid:
    """ move object on grid """
    return paint(cover(grid, obj), shift(obj, offset))

def tophalf(grid: Grid) -> Grid:
    """ upper half of grid """
    return grid[:len(grid) // 2]

def bottomhalf(grid: Grid) -> Grid:
    """ lower half of grid """
    return grid[len(grid) // 2 + len(grid) % 2:]

def lefthalf(grid: Grid) -> Grid:
    """ left half of grid """
    return rot270(tophalf(rot90(grid)))

def righthalf(grid: Grid) -> Grid:
    """ right half of grid """
    return rot270(bottomhalf(rot90(grid)))

def compress(grid: Grid) -> Grid:
    """ removes frontiers from grid """
    ri = tuple((i for i, r in enumerate(grid) if len(set(r)) == 1))
    ci = tuple((j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1))
    return tuple((tuple((v for j, v in enumerate(r) if j not in ci)) for i, r in enumerate(grid) if i not in ri))

