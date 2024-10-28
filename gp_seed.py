import random
import operator
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp, algorithms
from typing import Callable, Any
import inspect
import json
import dsl  # Ensure dsl.py is in the same directory or PYTHONPATH
import deterministic_utils as du  # Ensure deterministic_utils.py is available

# ===============================
# Define the Primitive Set
# ===============================

pset = gp.PrimitiveSetTyped("MAIN", [int, int], int)
pset.renameArguments(ARG0='x', ARG1='y')

# ===============================
# Define Safe Wrapper for Primitives
# ===============================

def safe_wrap(func: Callable) -> Callable:
    """Wraps a function to handle exceptions."""
    def wrapper(*args):
        try:
            result = func(*args)
            return int(result) if isinstance(result, bool) else result
        except Exception:
            return 0
    return wrapper

# ===============================
# Add DSL and Utility Primitives
# ===============================

for module in (dsl, du):
    for name, func in inspect.getmembers(module, inspect.isfunction):
        sig = inspect.signature(func)
        ret_type = sig.return_annotation
        if ret_type is int:
            arg_types = [int] * len(sig.parameters)
            wrapped_func = safe_wrap(func)
            pset.addPrimitive(wrapped_func, arg_types, int, name=name)

# Additional Primitives and Terminals
def if_then_else(condition: bool, output1: int, output2: int) -> int:
    return output1 if condition else output2

pset.addPrimitive(if_then_else, [bool, int, int], int, name="if_then_else")
pset.addPrimitive(operator.lt, [int, int], bool, name="less_than")
pset.addPrimitive(operator.eq, [int, int], bool, name="equal_to")
pset.addPrimitive(operator.gt, [int, int], bool, name="greater_than")
pset.addPrimitive(operator.and_, [bool, bool], bool, name="logical_and")
pset.addPrimitive(operator.or_, [bool, bool], bool, name="logical_or")
pset.addPrimitive(operator.not_, [bool], bool, name="logical_not")

pset.addTerminal(True, bool, name="True")
pset.addTerminal(False, bool, name="False")

# Random list generator for list-type terminals
def generate_random_list() -> list:
    return [random.randint(0, 10) for _ in range(random.randint(1, 5))]

pset.addTerminal(generate_random_list, list, name="generate_random_list")

# ===============================
# Set Up the Toolbox
# ===============================

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)  # Increased max depth
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# ===============================
# Strict Fitness and Evaluation Functions
# ===============================

def compute_strict_fitness(generated_grid: Any, target_grid: Any) -> float:
    """Calculates a strict Mean Squared Error with additional penalty."""
    mse = sum((cell_gen - cell_tar) ** 2
              for row_gen, row_tar in zip(generated_grid, target_grid)
              for cell_gen, cell_tar in zip(row_gen, row_tar))
    penalty = sum(1 for row_gen, row_tar in zip(generated_grid, target_grid)
                  for cell_gen, cell_tar in zip(row_gen, row_tar) if cell_gen != cell_tar)
    strict_fitness = mse / (len(target_grid) * len(target_grid[0])) + penalty
    return strict_fitness

def eval_individual(individual: creator.Individual) -> tuple:
    """Evaluates an individual based on a target grid."""
    func: Callable[[int, int], int] = toolbox.compile(expr=individual)
    target_grid = getattr(individual, 'target_grid', TARGET_GRID)
    grid_size = len(target_grid)
    generated_grid = tuple(tuple(func(x, y) for y in range(grid_size)) for x in range(grid_size))
    fitness = compute_strict_fitness(generated_grid, target_grid)
    return (fitness,)

toolbox.register("evaluate", eval_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# ===============================
# Population Initialization with Seeded Programs and Grids
# ===============================

def initialize_population(population_size, seed_file, seed_fraction=0.2):
    """Initializes a population with seeded programs and grids."""
    pop = []
    with open(seed_file, 'r') as f:
        seeds = json.load(f)

    seed_sample_size = int(population_size * seed_fraction)
    selected_seeds = random.sample(list(seeds.items()), min(seed_sample_size, len(seeds)))
    
    for expr_id, seed_data in selected_seeds:
        program_str = seed_data.get('program')
        input_grid = seed_data.get('input')
        output_grid = seed_data.get('output')
        
        if isinstance(program_str, str):
            try:
                individual = creator.Individual(gp.PrimitiveTree.from_string(program_str, pset))
                individual.input_grid = input_grid
                individual.target_grid = output_grid
                pop.append(individual)
            except Exception as e:
                print(f"Failed to parse expression for ID '{expr_id}': {e}")

    pop += [toolbox.individual() for _ in range(population_size - len(pop))]
    return pop

# ===============================
# Main Evolutionary Loop
# ===============================
POPULATION_SIZE = 100
GENERATIONS = 500
CX_PROB = 0.8  # Increased to favor crossover
MUT_PROB = 0.8  # Increased mutation rate for higher diversity

def visualize_grid(grid: Any, title: str = "Grid", fig_size: tuple = (5, 5)):
    """Visualizes a grid using matplotlib."""
    try:
        data = [[cell if isinstance(cell, int) else 0 for cell in row] for row in grid]
        fig, ax = plt.subplots(figsize=fig_size)
        ax.imshow(data, cmap='tab10', vmin=0, vmax=9)
        ax.set_xticks(np.arange(-0.5, len(data[0]), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(data), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        plt.show()
    except Exception as e:
        print(f"Error in visualize_grid: {e}")

# ===============================
# Define the Target Grid
# ===============================

TARGET_GRID = (
    (dsl.ZERO, dsl.ONE, dsl.ONE, dsl.ZERO, dsl.ZERO),
    (dsl.ONE, dsl.TWO, dsl.TWO, dsl.ONE, dsl.ZERO),
    (dsl.ONE, dsl.TWO, dsl.TWO, dsl.ONE, dsl.ZERO),
    (dsl.ZERO, dsl.ONE, dsl.ONE, dsl.ZERO, dsl.ZERO),
    (dsl.ZERO, dsl.ZERO, dsl.ZERO, dsl.ZERO, dsl.ZERO),
)

def main():
    random.seed(42)
    pop = initialize_population(POPULATION_SIZE, 'generated_grids.json')
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    mstats.register("std", np.std)
    
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
    
    func = toolbox.compile(expr=hof[0])
    try:
        grid_size = len(TARGET_GRID)
        generated_grid = tuple(
            tuple(func(x, y) for y in range(grid_size))
            for x in range(grid_size)
        )
        for row in generated_grid:
            print(row)
        
        visualize_grid(generated_grid, title="Best Generated Grid")
        visualize_grid(TARGET_GRID, title="Target Grid")
    except Exception as e:
        print(f"Error executing the best individual: {e}")

# Execute the Program
if __name__ == "__main__":
    main()

