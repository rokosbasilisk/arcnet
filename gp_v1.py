import random
import operator
import numpy as np
from deap import base, creator, tools, gp
from typing import Callable, Any, Tuple, List, Dict
import inspect
import json
from functools import partial
import sys

# ===============================
# Logo-Inspired DSL Primitives
# ===============================

# Define the turtle state as a dictionary
def initialize_turtle():
    return {
        'position': (15, 15),   # Start at center of a 30x30 grid
        'direction': 'N',       # N, E, S, W
        'pen': False,           # Pen up
        'color': 1,             # Initial color
        'grid': tuple([tuple([0]*30) for _ in range(30)])  # 30x30 grid initialized to 0
    }

# Movement Commands
def forward(turtle: Dict[str, Any], steps: int) -> Dict[str, Any]:
    x, y = turtle['position']
    direction = turtle['direction']
    grid = [list(row) for row in turtle['grid']]

    for _ in range(steps):
        if direction == 'N':
            x = max(0, x - 1)
        elif direction == 'S':
            x = min(29, x + 1)
        elif direction == 'E':
            y = min(29, y + 1)
        elif direction == 'W':
            y = max(0, y - 1)

        if turtle['pen']:
            grid[x][y] = turtle['color']

        turtle['position'] = (x, y)

    turtle['grid'] = tuple(tuple(row) for row in grid)
    return turtle

def turn_left(turtle: Dict[str, Any]) -> Dict[str, Any]:
    directions = ['N', 'W', 'S', 'E']
    idx = directions.index(turtle['direction'])
    turtle['direction'] = directions[(idx + 1) % 4]
    return turtle

def turn_right(turtle: Dict[str, Any]) -> Dict[str, Any]:
    directions = ['N', 'E', 'S', 'W']
    idx = directions.index(turtle['direction'])
    turtle['direction'] = directions[(idx + 1) % 4]
    return turtle

# Pen Commands
def pen_up(turtle: Dict[str, Any]) -> Dict[str, Any]:
    turtle['pen'] = False
    return turtle

def pen_down(turtle: Dict[str, Any]) -> Dict[str, Any]:
    turtle['pen'] = True
    return turtle

# Drawing Commands
def set_color(turtle: Dict[str, Any], color: int) -> Dict[str, Any]:
    turtle['color'] = max(0, min(color, 9))  # Clamp color to [0,9]
    return turtle

# Control Structures
def if_then_else(turtle: Dict[str, Any], condition: bool, true_func: Callable, false_func: Callable) -> Dict[str, Any]:
    if condition:
        return true_func(turtle)
    else:
        return false_func(turtle)

def loop(turtle: Dict[str, Any], iterations: int, func: Callable) -> Dict[str, Any]:
    for _ in range(iterations):
        turtle = func(turtle)
    return turtle

# No-Operation Function
def noop(turtle: Dict[str, Any]) -> Dict[str, Any]:
    return turtle

# ===============================
# Define the Primitive Set
# ===============================

# Initialize the primitive set with turtle state as input and output
pset = gp.PrimitiveSetTyped("MAIN", [dict], dict)
pset.renameArguments(ARG0='turtle')

# Add primitives
pset.addPrimitive(forward, [dict, int], dict, name="forward")
pset.addPrimitive(turn_left, [dict], dict, name="turn_left")
pset.addPrimitive(turn_right, [dict], dict, name="turn_right")
pset.addPrimitive(pen_up, [dict], dict, name="pen_up")
pset.addPrimitive(pen_down, [dict], dict, name="pen_down")
pset.addPrimitive(set_color, [dict, int], dict, name="set_color")
pset.addPrimitive(if_then_else, [dict, bool, Callable, Callable], dict, name="if_then_else")
pset.addPrimitive(loop, [dict, int, Callable], dict, name="loop")
pset.addPrimitive(noop, [dict], dict, name="noop")

# Add terminals
for i in range(10):
    pset.addTerminal(i, int, name=str(i))
pset.addTerminal(True, bool, name="True")
pset.addTerminal(False, bool, name="False")

# ===============================
# Initialize DEAP Components
# ===============================

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizing fitness
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Define how to generate expressions
toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

# Define individual and population
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define how to compile expressions
toolbox.register("compile", gp.compile, pset=pset)

# ===============================
# Define Fitness Function
# ===============================

def pad_grid(grid: List[List[int]], target_rows: int, target_cols: int, pad_value: int = 0) -> Tuple[Tuple[int]]:
    """Pads the grid to the target size with the pad_value."""
    padded_grid = []
    for row in grid:
        if len(row) < target_cols:
            padded_row = row + [pad_value] * (target_cols - len(row))
        else:
            padded_row = row[:target_cols]
        padded_grid.append(padded_row)
    # Add additional rows if necessary
    while len(padded_grid) < target_rows:
        padded_grid.append([pad_value] * target_cols)
    # Trim extra rows if necessary
    padded_grid = padded_grid[:target_rows]
    return tuple(tuple(row) for row in padded_grid)

def execute_program(individual: Any, target_grid: Tuple[Tuple[int]], initial_turtle: Dict[str, Any]) -> float:
    """Executes the individual's program and computes fitness based on the difference from target grid."""
    try:
        func = toolbox.compile(expr=individual)
        turtle = initial_turtle.copy()
        turtle = func(turtle)
        generated_grid = turtle['grid']
        mse = 0.0
        for row_gen, row_tar in zip(generated_grid, target_grid):
            for cell_gen, cell_tar in zip(row_gen, row_tar):
                mse += (cell_gen - cell_tar) ** 2
        total_cells = len(target_grid) * len(target_grid[0])
        return mse / total_cells
    except Exception as e:
        print(f"Execution error: {e}")
        return float('inf')

def eval_individual(individual: Any, training_samples: List[Dict[str, Any]]) -> Tuple[float,]:
    """Evaluates an individual based on multiple training samples."""
    total_fitness = 0.0
    for idx, sample in enumerate(training_samples):
        input_grid = sample['input']
        target_grid = sample['output']
        # Initialize turtle
        initial_turtle = initialize_turtle()
        # Set the grid to input_grid
        initial_turtle['grid'] = pad_grid(input_grid, 30, 30)
        # Execute program
        fitness = execute_program(individual, pad_grid(target_grid, 30, 30), initial_turtle)
        total_fitness += fitness
    # Average fitness over samples
    return (total_fitness / len(training_samples),)

# Register the fitness function
toolbox.register("evaluate", eval_individual, training_samples=[])

# ===============================
# Register Genetic Operators
# ===============================

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=gp.genFull, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

# ===============================
# Load and Prepare Dataset
# ===============================

def load_dataset(seed_file: str, max_samples: int = 100) -> List[Dict[str, Any]]:
    """Loads the dataset from a JSON file."""
    try:
        with open(seed_file, 'r') as f:
            data = json.load(f)
        # Assuming data is a list of challenges with 'input' and 'output'
        samples = []
        for challenge in data:
            input_grid = challenge.get('input')
            output_grid = challenge.get('output')
            if input_grid and output_grid:
                samples.append({'input': input_grid, 'output': output_grid})
                if len(samples) >= max_samples:
                    break
        return samples
    except FileNotFoundError:
        print(f"Seed file '{seed_file}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from '{seed_file}'. Ensure it's properly formatted.")
        return []

# ===============================
# Visualization Function
# ===============================

def colorize(value: int) -> str:
    """Converts a cell value to a colored string for terminal display."""
    color_map = {
        0: "\033[40m  \033[0m",  # Black
        1: "\033[41m  \033[0m",  # Red
        2: "\033[42m  \033[0m",  # Green
        3: "\033[43m  \033[0m",  # Yellow
        4: "\033[44m  \033[0m",  # Blue
        5: "\033[45m  \033[0m",  # Magenta
        6: "\033[46m  \033[0m",  # Cyan
        7: "\033[47m  \033[0m",  # White
        8: "\033[100m  \033[0m", # Gray
        9: "\033[101m  \033[0m", # Bright Red
    }
    return color_map.get(value, "\033[107m  \033[0m")  # Default to bright white

def display_grids(generated: Tuple[Tuple[int]], target: Tuple[Tuple[int]]):
    """Displays generated and target grids side by side."""
    print("\nGenerated Grid vs. Target Grid:")
    print("Generated:".ljust(15) + "Target:")
    for row_gen, row_tar in zip(generated, target):
        gen_str = "".join(colorize(cell) for cell in row_gen)
        tar_str = "".join(colorize(cell) for cell in row_tar)
        print(gen_str + "    " + tar_str)
    print("\033[0m")  # Reset terminal colors

# ===============================
# Main Evolutionary Loop
# ===============================

def main():
    # Parameters
    POPULATION_SIZE = 100
    GENERATIONS = 50
    CX_PROB = 0.5  # Crossover probability
    MUT_PROB = 0.2  # Mutation probability

    # Load dataset
    SEED_FILE = 'data/arc-agi_training_challenges.json'  # Update this path as needed
    training_samples = load_dataset(SEED_FILE, max_samples=100)
    if not training_samples:
        print("No training samples available. Exiting.")
        return
    print(f"Loaded {len(training_samples)} training samples.")

    # Update the evaluate function with actual training samples
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", eval_individual, training_samples=training_samples)

    # Initialize population
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)  # Keep track of the best individual
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    # Evolutionary Loop
    for gen in range(1, GENERATIONS + 1):
        print(f"=== Generation {gen} ===")
        
        # Select and clone the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print(f"  Evaluating {len(invalid_ind)} individuals...")
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population with offspring
        pop[:] = offspring
        
        # Update the hall of fame with the best individual
        hof.update(pop)
        
        # Gather all the fitnesses in one list and print the stats
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        
        # Optional: Print the best individual every 10 generations
        if gen % 10 == 0:
            best = hof[0]
            print(f"Best Individual at Generation {gen}:")
            print(best)
            # Optionally, visualize the best individual's output against a sample
            sample = random.choice(training_samples)
            input_grid = sample['input']
            target_grid = sample['output']
            # Initialize turtle with input grid
            initial_turtle = initialize_turtle()
            initial_turtle['grid'] = pad_grid(input_grid, 30, 30)
            try:
                func = toolbox.compile(expr=best)
                turtle = func(initial_turtle)
                generated_grid = turtle['grid']
                # Compute fitness
                fitness = execute_program(best, pad_grid(target_grid, 30, 30), initial_turtle)
                print(f"Fitness: {fitness:.4f}")
                # Visualize grids
                display_grids(generated_grid, pad_grid(target_grid, 30, 30))
            except Exception as e:
                print(f"Error executing best individual: {e}")

    # Final evaluation of all individuals
    print("=== Final Evaluation ===")
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    print(f"  Evaluating {len(invalid_ind)} individuals...")
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)

    print("\n=== Best Individual ===")
    best = hof[0]
    print(best)
    print(f"Fitness: {best.fitness.values[0]:.4f}")

    # Visualize final best individual
    sample = random.choice(training_samples)
    input_grid = sample['input']
    target_grid = sample['output']
    # Initialize turtle with input grid
    initial_turtle = initialize_turtle()
    initial_turtle['grid'] = pad_grid(input_grid, 30, 30)
    try:
        func = toolbox.compile(expr=best)
        turtle = func(initial_turtle)
        generated_grid = turtle['grid']
        # Compute fitness
        fitness = execute_program(best, pad_grid(target_grid, 30, 30), initial_turtle)
        print(f"Final Fitness: {fitness:.4f}")
        # Visualize grids
        display_grids(generated_grid, pad_grid(target_grid, 30, 30))
    except Exception as e:
        print(f"Error executing best individual: {e}")

# ===============================
# Entry Point
# ===============================

if __name__ == "__main__":
    main()

