import random
import operator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deap import base, creator, tools, gp
from typing import Callable, Any
import inspect
import json
from functools import partial
from re_arc import dsl  # Ensure re_arc/dsl.py is in the 're_arc' directory
import re_arc.deterministic_utils as du  # Ensure re_arc/deterministic_utils.py is available
import os

# ===============================
# Neural Network for Guiding Primitive Selection
# ===============================

class PrimitiveSelector(nn.Module):
    def __init__(self, input_size, output_size):
        super(PrimitiveSelector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

# ===============================
# Define the Primitive Set with Enhanced Conditionals
# ===============================

pset = gp.PrimitiveSetTyped("MAIN", [int, int, list], int)
pset.renameArguments(ARG0='x', ARG1='y', ARG2='input_grid')

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

# ===============================
# Add Enhanced Conditionals and Logical Primitives
# ===============================

def if_then_else(condition: bool, output1: int, output2: int) -> int:
    return output1 if condition else output2

pset.addPrimitive(if_then_else, [bool, int, int], int, name="if_then_else")
pset.addPrimitive(operator.lt, [int, int], bool, name="less_than")
pset.addPrimitive(operator.eq, [int, int], bool, name="equal_to")
pset.addPrimitive(operator.gt, [int, int], bool, name="greater_than")
pset.addPrimitive(operator.and_, [bool, bool], bool, name="logical_and")
pset.addPrimitive(operator.or_, [bool, bool], bool, name="logical_or")
pset.addPrimitive(operator.not_, [bool], bool, name="logical_not")

# Add boolean terminals
pset.addTerminal(True, bool, name="True")
pset.addTerminal(False, bool, name="False")

# Random list generator for list-type terminals
def generate_random_list() -> list:
    """Generates a random list of integers."""
    return [random.randint(0, 10) for _ in range(random.randint(1, 5))]

pset.addTerminal(generate_random_list, list, name="generate_random_list")

# ===============================
# Diagnostic: Inspect pset.primitives
# ===============================

print("Primitives in pset:")
for prim in pset.primitives:
    if hasattr(prim, 'name'):
        print(f" - {prim.name}")
    else:
        print(f" - {prim} (No name attribute)")

# ===============================
# Initialize DEAP Components
# ===============================

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# ===============================
# Define Fitness Function: Overlapping Grids
# ===============================

def pad_grid(grid, target_rows, target_cols, pad_value=0):
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
    return padded_grid

def compute_overlap_fitness(generated_grid, target_grid):
    """Calculates fitness based on the number of overlapping cells."""
    if generated_grid is None:
        return float('inf')
    try:
        overlap = sum(
            1 for row_gen, row_tar in zip(generated_grid, target_grid)
            for cell_gen, cell_tar in zip(row_gen, row_tar)
            if cell_gen == cell_tar and cell_gen != 0
        )
        total = len(target_grid) * len(target_grid[0])
        return (total - overlap) / total  # Lower is better
    except TypeError:
        return float('inf')

def eval_individual(individual, training_samples):
    """Evaluates an individual based on grid overlap over training samples."""
    total_fitness = 0.0
    samples_evaluated = 0
    try:
        func = toolbox.compile(expr=individual)
        for sample in training_samples:
            input_grid = sample['input']
            target_grid = sample['output']
            # Enforce fixed grid size of 30x30 by padding if necessary
            input_grid_padded = pad_grid(input_grid, 30, 30)
            target_grid_padded = pad_grid(target_grid, 30, 30)
            # Generate the grid using the individual
            generated_grid = [[func(x, y, input_grid_padded) for y in range(30)] for x in range(30)]
            # Clamp generated values to [0, 9]
            generated_grid = [
                [max(0, min(cell, 9)) if isinstance(cell, int) else 0 for cell in row]
                for row in generated_grid
            ]
            # Compute fitness as overlap measure
            fitness = compute_overlap_fitness(generated_grid, target_grid_padded)
            total_fitness += fitness
            samples_evaluated += 1
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        return (float('inf'),)
    if samples_evaluated == 0:
        return (float('inf'),)
    # Return average fitness over samples
    return (total_fitness / samples_evaluated,)

toolbox.register("evaluate", eval_individual, training_samples=[])  # Placeholder, will set later
toolbox.register("mate", gp.cxOnePoint)

# ===============================
# Mutation Function Registration
# ===============================

mutate_expr = partial(gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=mutate_expr, pset=pset)

# ===============================
# Visualization Function with Terminal Colors
# ===============================

def colorize(value):
    color_map = {
        0: "\033[40m  \033[0m",  # Black
        1: "\033[41m  \033[0m",  # Red
        2: "\033[42m  \033[0m",  # Green
        3: "\033[43m  \033[0m",  # Yellow
        4: "\033[44m  \033[0m",  # Blue
        5: "\033[45m  \033[0m",  # Magenta
        6: "\033[46m  \033[0m",  # Cyan
        7: "\033[47m  \033[0m",  # White
        8: "\033[100m  \033[0m",  # Gray
        9: "\033[101m  \033[0m",  # Bright Red
    }
    return color_map.get(value, "\033[107m  \033[0m")  # Default to bright white

def display_colored_grid(grid, title="Grid"):
    print(f"{title}:")
    for row in grid:
        print("".join(colorize(cell) for cell in row))
    print("\033[0m")  # Reset colors

# ===============================
# Main Function
# ===============================

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Primitive Selector Network
    primitive_list = [prim for prim in pset.primitives if hasattr(prim, 'name')]
    primitive_count = len(primitive_list)
    network = PrimitiveSelector(input_size=primitive_count, output_size=primitive_count)
    network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Load and prepare dataset for GP training
    SEED_FILE = 'data/arc-agi_training_challenges.json'
    TRAIN_SIZE_NET = 80
    VAL_SIZE_NET = 20

    print("Loading dataset for GP training...")
    try:
        with open(SEED_FILE, 'r') as f:
            seeds = json.load(f)
    except FileNotFoundError:
        print(f"Seed file '{SEED_FILE}' not found.")
        return

    all_train_samples = []
    for data in seeds.values():
        all_train_samples.extend(data.get('train', []))
    # Filter samples with grid sizes <=30x30
    valid_train_samples = [
        sample for sample in all_train_samples
        if 'input' in sample and 'output' in sample
        and len(sample['input']) <= 30 and len(sample['output']) <= 30
        and all(len(row_input) <= 30 for row_input in sample['input'])
        and all(len(row_output) <= 30 for row_output in sample['output'])
    ]
    if len(valid_train_samples) < TRAIN_SIZE_NET + VAL_SIZE_NET:
        print(f"Warning: Not enough valid training samples. Required: {TRAIN_SIZE_NET + VAL_SIZE_NET}, Available: {len(valid_train_samples)}")
    # Shuffle and split
    random.shuffle(valid_train_samples)
    training_samples_gp = valid_train_samples[:TRAIN_SIZE_NET]
    validation_samples_gp = valid_train_samples[TRAIN_SIZE_NET:TRAIN_SIZE_NET + VAL_SIZE_NET]
    print(f"Using {len(training_samples_gp)} training samples for GP.")

    # Initialize population
    POPULATION_SIZE = 200
    GENERATIONS = 100
    CX_PROB = 0.5  # Crossover probability
    MUT_PROB = 0.2  # Mutation probability

    pop = toolbox.population(n=POPULATION_SIZE)
    print(f"Initialized population with {len(pop)} individuals.")

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    # Initialize primitive usage frequency
    primitive_usage = {prim.name: 0 for prim in primitive_list}

    # Update the toolbox evaluate function with the training samples
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", eval_individual, training_samples=training_samples_gp)

    # Evolutionary Loop
    for gen in range(1, GENERATIONS + 1):
        print(f"=== Generation {gen} ===")

        # Evaluate individuals that have not yet been evaluated
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        print(f"  Evaluating {len(invalid_ind)} individuals...")
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update primitive usage statistics
        for ind in invalid_ind:
            for node in ind:
                if hasattr(node, 'name') and node.name in primitive_usage:
                    primitive_usage[node.name] += 1

        # Prepare data for neural network training
        usage_vectors = []
        fitness_scores = []
        for ind in invalid_ind:
            # Create a binary vector indicating the usage of each primitive in the individual
            usage_vector = [0] * primitive_count
            for node in ind:
                if hasattr(node, 'name'):
                    if node.name in primitive_usage:
                        index = primitive_list.index(next(prim for prim in primitive_list if prim.name == node.name))
                        usage_vector[index] += 1
            usage_vectors.append(usage_vector)
            fitness_scores.append(ind.fitness.values[0])

        if usage_vectors and fitness_scores:
            usage_tensor = torch.tensor(usage_vectors, dtype=torch.float32).to(device)
            fitness_tensor = torch.tensor(fitness_scores, dtype=torch.float32).unsqueeze(1).to(device)
            optimizer.zero_grad()
            predictions = network(usage_tensor)
            loss = loss_fn(predictions, fitness_tensor)
            loss.backward()
            optimizer.step()
            print(f"  Trained Primitive Selector Network with loss: {loss.item():.4f}")

        # Use the network to get primitive selection probabilities
        if any(primitive_usage.values()):
            usage_vector = [primitive_usage[prim.name] for prim in primitive_list]
            usage_vector = np.array(usage_vector, dtype=np.float32)
            usage_vector = usage_vector / usage_vector.sum() if usage_vector.sum() > 0 else np.ones_like(usage_vector) / len(usage_vector)
            usage_tensor = torch.tensor([usage_vector], dtype=torch.float32).to(device)
            with torch.no_grad():
                selection_probs = network(usage_tensor).cpu().numpy()[0]
            print(f"  Primitive selection probabilities: {selection_probs}")

        # Update Hall of Fame
        hof.update(pop)

        # Gather all fitnesses for statistics
        fits = [ind.fitness.values[0] for ind in pop]

        # Record statistics
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

        # **Periodic Reporting Every 10 Generations**
        if gen % 10 == 0:
            print(f"\n--- Reporting at Generation {gen} ---")
            best_individual = hof[0]
            print(f"Best Individual:\n{best_individual}")
            
            # Compile the best individual
            func = toolbox.compile(expr=best_individual)
            
            try:
                # Select a sample from validation for visualization
                if len(validation_samples_gp) > 0:
                    sample = validation_samples_gp[0]
                    input_grid = sample['input']
                    target_grid = sample['output']
                    
                    # Enforce fixed grid size of 30x30 by padding if necessary
                    input_grid_padded = pad_grid(input_grid, 30, 30)
                    target_grid_padded = pad_grid(target_grid, 30, 30)
                    
                    # Generate the grid using the best individual
                    generated_grid = [[func(x, y, input_grid_padded) for y in range(30)] for x in range(30)]
                    
                    # Clamp generated values to [0, 9]
                    generated_grid = [
                        [max(0, min(cell, 9)) if isinstance(cell, int) else 0 for cell in row]
                        for row in generated_grid
                    ]
                    
                    # Display the grids
                    print("\n--- Best Individual's Generated Grid ---")
                    display_colored_grid(generated_grid, "Generated Grid")
                    
                    print("\n--- Target Grid ---")
                    display_colored_grid(target_grid_padded, "Target Grid")
                else:
                    print("No validation samples available for visualization.")
            except Exception as e:
                print(f"Error executing the best individual: {e}")
            print("--- End of Reporting ---\n")

        # Selection
        print("  Selecting individuals...")
        selected = tools.selTournament(pop, k=POPULATION_SIZE // 2, tournsize=3)
        offspring = [toolbox.clone(ind) for ind in selected]

        # Apply crossover and mutation on the offspring
        print("  Applying crossover and mutation...")
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # The population for the next generation is the offspring
        pop = offspring

    # Final evaluation of all individuals
    print("=== Final Evaluation ===")
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    print(f"  Evaluating {len(invalid_ind)} individuals...")
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)

    print("\nBest individual:", hof[0])

    # Compile and visualize the best individual
    func = toolbox.compile(expr=hof[0])
    try:
        if len(validation_samples_gp) > 0:
            sample = validation_samples_gp[0]
            input_grid = sample['input']
            target_grid = sample['output']
            # Enforce fixed grid size of 30x30 by padding if necessary
            input_grid_padded = pad_grid(input_grid, 30, 30)
            target_grid_padded = pad_grid(target_grid, 30, 30)
            # Generate the grid using the best individual
            generated_grid = [[func(x, y, input_grid_padded) for y in range(30)] for x in range(30)]
            # Clamp generated values to [0, 9]
            generated_grid = [
                [max(0, min(cell, 9)) if isinstance(cell, int) else 0 for cell in row]
                for row in generated_grid
            ]
            print("Generated Grid vs. Target Grid:")
            display_colored_grid(generated_grid, "Generated Grid")
            display_colored_grid(target_grid_padded, "Target Grid")
        else:
            print("No validation samples available for visualization.")
    except Exception as e:
        print(f"Error executing the best individual: {e}")

if __name__ == "__main__":
    main()

