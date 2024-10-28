import random
import operator
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp, algorithms
from typing import Callable, Any, get_origin, Union
import inspect
import dsl  # Ensure dsl.py is in the same directory or PYTHONPATH

# ===============================
# Define FunctionWrapper Class
# ===============================

class FunctionWrapper:
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, *args):
        return self.func(*args)

    def __repr__(self):
        return f"FunctionWrapper({self.func.__name__})"

# ===============================
# Define Fitness and Individual
# ===============================

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# ===============================
# Define the Primitive Set
# ===============================

# PrimitiveSetTyped: Name, [input types], output type
pset = gp.PrimitiveSetTyped("MAIN", [int, int], int)
pset.renameArguments(ARG0='x', ARG1='y')  # Optional: Rename arguments for clarity

# ===============================
# Define Safe Wrapper for Primitives
# ===============================

def safe_wrap(func: Callable) -> Callable:
    """
    Wraps a function to handle exceptions and ensure return types are consistent.
    """
    sig = inspect.signature(func)
    arity = len(sig.parameters)

    def wrapper(*args):
        try:
            result = func(*args)
            return int(result) if isinstance(result, bool) else result
        except Exception:
            return 0  # Default value if there's an error
    return wrapper

# ===============================
# Automate Adding Primitives from DSL
# ===============================

def is_concrete(annotation):
    """
    Determines if a type annotation is concrete (i.e., not a Union, Callable, Any, etc.).
    """
    origin = get_origin(annotation)
    return origin is None and annotation not in {Union, Callable, Any}

# Iterate over all functions in dsl module and add to pset
for name, func in inspect.getmembers(dsl, inspect.isfunction):
    sig = inspect.signature(func)
    ret_type = sig.return_annotation
    if ret_type is inspect.Signature.empty or not is_concrete(ret_type):
        continue

    arg_types = []
    skip = False
    for param in sig.parameters.values():
        param_type = param.annotation
        if param_type is inspect.Parameter.empty or not is_concrete(param_type):
            skip = True
            break
        arg_types.append(int)  # Assume all arguments are int

    if skip:
        continue  # Skip non-concrete types

    wrapped_func = safe_wrap(func)
    try:
        pset.addPrimitive(wrapped_func, arg_types, int, name=name)
    except TypeError as e:
        print(f"Failed to add primitive '{name}': {e}")

# ===============================
# Add Primitives for Conditionals and Loops
# ===============================

# Define if-then-else primitive
def if_then_else(condition: bool, output1: int, output2: int) -> int:
    return output1 if condition else output2

pset.addPrimitive(if_then_else, [bool, int, int], int, name="if_then_else")

# Add comparison operators with safe names
pset.addPrimitive(operator.lt, [int, int], bool, name="less_than")
pset.addPrimitive(operator.eq, [int, int], bool, name="equal_to")
pset.addPrimitive(operator.gt, [int, int], bool, name="greater_than")

# Add logical operators with renamed functions to avoid conflicts
pset.addPrimitive(operator.and_, [bool, bool], bool, name="logical_and")
pset.addPrimitive(operator.or_, [bool, bool], bool, name="logical_or")
pset.addPrimitive(operator.not_, [bool], bool, name="logical_not")

# Add boolean terminals
pset.addTerminal(True, bool, name="True")
pset.addTerminal(False, bool, name="False")

# Define loop-like primitive (e.g., sum over a range)
def sum_range(start: int, end: int) -> int:
    # Limit the range to prevent excessive computation
    start = max(0, min(start, 100))
    end = max(start, min(end, 100))
    return sum(range(start, end + 1))

pset.addPrimitive(sum_range, [int, int], int, name="sum_range")

# ===============================
# Add Terminals (Constants)
# ===============================

# Add color constants (assuming they are defined as ZERO, ONE, ..., NINE in dsl.py)
color_constants = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
for color_name in color_constants:
    if hasattr(dsl, color_name):
        color_value = getattr(dsl, color_name)
        pset.addTerminal(color_value, int, name=color_name)

# Add integer constants
for i in range(10):
    pset.addTerminal(i, int, name=f"const_{i}")

# Add identity function if not defined
def identity(x: int) -> int:
    return x

pset.addPrimitive(identity, [int], int, name="identity")

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

def compute_fitness(generated_grid: Any, target_grid: Any) -> float:
    """
    Computes the Mean Squared Error between the generated grid and the target grid.
    """
    try:
        mse = sum((cell_gen - cell_tar) ** 2
                  for row_gen, row_tar in zip(generated_grid, target_grid)
                  for cell_gen, cell_tar in zip(row_gen, row_tar))
        return mse / (len(target_grid) * len(target_grid[0]))
    except Exception as e:
        print(f"Error in compute_fitness: {e}")
        return float('inf')

# ===============================
# Evaluation Function
# ===============================

# Define the target grid
TARGET_GRID = (
    (dsl.ZERO, dsl.ONE, dsl.ONE, dsl.ZERO, dsl.ZERO),
    (dsl.ONE, dsl.TWO, dsl.TWO, dsl.ONE, dsl.ZERO),
    (dsl.ONE, dsl.TWO, dsl.TWO, dsl.ONE, dsl.ZERO),
    (dsl.ZERO, dsl.ONE, dsl.ONE, dsl.ZERO, dsl.ZERO),
    (dsl.ZERO, dsl.ZERO, dsl.ZERO, dsl.ZERO, dsl.ZERO),
)

def eval_individual(individual: creator.Individual) -> tuple:
    """
    Evaluates an individual by compiling its expression and computing its fitness.
    """
    func: Callable[[int, int], int] = toolbox.compile(expr=individual)
    try:
        grid_size = len(TARGET_GRID)
        generated_grid = tuple(
            tuple(func(x, y) for y in range(grid_size))
            for x in range(grid_size)
        )
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

POPULATION_SIZE = 100
GENERATIONS = 100
CX_PROB = 0.5
MUT_PROB = 0.3

# ===============================
# Visualization Function
# ===============================

def visualize_grid(grid: Any, title: str = "Grid", fig_size: tuple = (5, 5)):
    """
    Visualizes the grid using matplotlib.
    """
    try:
        data = [[cell if isinstance(cell, int) else 0 for cell in row] for row in grid]
        h, w = len(data), len(data[0]) if data else 0
        fig, ax = plt.subplots(figsize=fig_size)
        ax.imshow(data, cmap='tab10', vmin=0, vmax=9)
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        plt.show()
    except Exception as e:
        print(f"Error in visualize_grid: {e}")

# ===============================
# Main Evolutionary Loop
# ===============================

def main():
    """
    Executes the main evolutionary loop.
    """
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
    
    # Add verbosity to track progress
    print("Starting evolution...")
    
    pop, log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        ngen=GENERATIONS,
        stats=mstats,
        halloffame=hof,
        verbose=True
    )
    
    print("\nBest individual:")
    print(hof[0])
    
    func: Callable[[int, int], int] = toolbox.compile(expr=hof[0])
    try:
        grid_size = len(TARGET_GRID)
        generated_grid = tuple(
            tuple(func(x, y) for y in range(grid_size))
            for x in range(grid_size)
        )
        print("\nGenerated Grid:")
        for row in generated_grid:
            print(row)
        
        # Visualize the best generated grid and the target grid
        visualize_grid(generated_grid, title="Best Generated Grid")
        visualize_grid(TARGET_GRID, title="Target Grid")
    except Exception as e:
        print(f"Error executing the best individual: {e}")

# Execute the Program
if __name__ == "__main__":
    main()

