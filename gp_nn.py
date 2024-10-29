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
import functools
from functools import partial
import dsl  # Ensure dsl.py is in the same directory or PYTHONPATH
import deterministic_utils as du  # Ensure deterministic_utils.py is available

# ===============================
# Neural Network Model to Predict Fitness
# ===============================

class FitnessPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super(FitnessPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ===============================
# Define the Primitive Set
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
# Add Additional Primitives and Terminals
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
toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# ===============================
# Define Fitness Function
# ===============================

def compute_strict_fitness(generated_grid, target_grid):
    """Calculates a strict fitness score with MSE and penalty for mismatches."""
    if generated_grid is None:
        return float('inf')
    try:
        mse = sum((cell_gen - cell_tar) ** 2 for row_gen, row_tar in zip(generated_grid, target_grid)
                  for cell_gen, cell_tar in zip(row_gen, row_tar))
        penalty = sum(1 for row_gen, row_tar in zip(generated_grid, target_grid)
                      for cell_gen, cell_tar in zip(row_gen, row_tar) if cell_gen != cell_tar)
        return mse + penalty
    except TypeError:
        return float('inf')

def eval_individual(individual, training_samples):
    """Evaluates an individual based on multiple training samples."""
    total_fitness = 0.0
    samples_evaluated = 0
    try:
        func = toolbox.compile(expr=individual)
        for sample in training_samples:
            input_grid = sample['input']
            target_grid = sample['output']
            grid_size = len(input_grid)
            if len(target_grid) != grid_size or any(len(row) != len(input_grid[0]) for row in target_grid):
                continue
            generated_grid = [[func(x, y, input_grid) for y in range(len(input_grid[0]))] for x in range(grid_size)]
            fitness = compute_strict_fitness(generated_grid, target_grid)
            total_fitness += fitness
            samples_evaluated += 1
    except Exception:
        return (float('inf'),)
    if samples_evaluated == 0:
        return (float('inf'),)
    return (total_fitness,)

toolbox.register("evaluate", eval_individual, training_samples=[])  # Placeholder, will set later
toolbox.register("mate", gp.cxOnePoint)

# ===============================
# Mutation Function Registration
# ===============================

mutate_expr = partial(gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=mutate_expr, pset=pset)

# ===============================
# Convert Individual to Feature Vector
# ===============================

def individual_to_features(individual):
    """Convert the individual program into a feature vector for prediction."""
    depth = individual.height
    num_nodes = len(individual)
    terminals = sum(1 for node in individual if isinstance(node, gp.Terminal))
    primitives = num_nodes - terminals
    primitive_names = [prim.name for prim in pset.primitives if hasattr(prim, 'name')]
    primitive_counts = {name: 0 for name in primitive_names}
    for node in individual:
        if isinstance(node, gp.Primitive):
            primitive_counts[node.name] += 1
    primitive_features = [primitive_counts[name] for name in primitive_names]
    features = [depth, num_nodes, terminals, primitives] + primitive_features
    if len(features) < 20:
        features += [0] * (20 - len(features))
    else:
        features = features[:20]
    return features

# ===============================
# Neural Network for Fitness Prediction
# ===============================

model = FitnessPredictor(input_size=20)  # Adjust input_size if more features are added
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

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
# Main Evolutionary Loop
# ===============================

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    POPULATION_SIZE = 200
    GENERATIONS = 100
    CX_PROB = 0.5
    MUT_PROB = 0.2
    SEED_FILE = 'arc-agi_training_challenges.json'
    TRAIN_SIZE = 80
    VAL_SIZE = 20
    
    train_seeds, val_seeds = load_dataset(SEED_FILE, train_size=TRAIN_SIZE, val_size=VAL_SIZE)
    pop = initialize_population(POPULATION_SIZE, training_samples=train_seeds, seed_fraction=0.2)
    hof = tools.HallOfFame(1)

    nn_training_features = []
    nn_training_fitnesses = []

    for gen in range(1, GENERATIONS + 1):
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = [eval_individual(ind, train_seeds)[0] for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
            nn_training_features.append(individual_to_features(ind))
            nn_training_fitnesses.append(fit)
        
        hof.update(pop)

        if nn_training_features:
            features_tensor = torch.tensor(nn_training_features, dtype=torch.float32)
            fitness_tensor = torch.tensor(nn_training_fitnesses, dtype=torch.float32).unsqueeze(1)
            train_predictor(model, optimizer, loss_fn, features_tensor, fitness_tensor, epochs=10)

        selected = toolbox.select_guided(pop, k=POPULATION_SIZE // 2)
        offspring = [toolbox.clone(ind) for ind in selected]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        pop = offspring

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = [eval_individual(ind, train_seeds)[0] for ind in invalid_ind]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = (fit,)
        nn_training_features.append(individual_to_features(ind))
        nn_training_fitnesses.append(fit)

    hof.update(pop)

    func = toolbox.compile(expr=hof[0])
    try:
        if val_seeds:
            sample = val_seeds[0]
            input_grid = sample['input']
            target_grid = sample['output']
            generated_grid = [[func(x, y, input_grid) for y in range(len(input_grid[0]))] for x in range(len(input_grid))]
            print("Generated Grid vs. Target Grid:")
            display_colored_grid(generated_grid, "Generated Grid")
            display_colored_grid(target_grid, "Target Grid")
        else:
            print("No validation samples available.")
    except Exception as e:
        print(f"Error executing the best individual: {e}")

if __name__ == "__main__":
    main()

