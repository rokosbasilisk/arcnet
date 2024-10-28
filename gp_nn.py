import random
import operator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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
        # If generated_grid structure doesn't match target_grid
        return float('inf')

def eval_individual(individual):
    """Evaluates an individual based on the target grid."""
    try:
        func = toolbox.compile(expr=individual)
        grid_size = len(TARGET_GRID)
        generated_grid = tuple(tuple(func(x, y) for y in range(grid_size)) for x in range(grid_size))
        fitness = compute_strict_fitness(generated_grid, TARGET_GRID)
        return (fitness,)
    except Exception as e:
        # Assign a high fitness value if execution fails
        print(f"Error evaluating individual: {e}")
        return (float('inf'),)

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", gp.cxOnePoint)

# ===============================
# Mutation Function Registration
# ===============================

# Use functools.partial to pass fixed parameters to gp.genFull
mutate_expr = partial(gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=mutate_expr, pset=pset)

# ===============================
# Define the Target Grid
# ===============================

TARGET_GRID = (
    (0, 1, 1, 0, 0),
    (1, 2, 2, 1, 0),
    (1, 2, 2, 1, 0),
    (0, 1, 1, 0, 0),
    (0, 0, 0, 0, 0),
)

# ===============================
# Convert Individual to Feature Vector
# ===============================

def individual_to_features(individual):
    """Convert the individual program into a feature vector for prediction."""
    depth = individual.height
    num_nodes = len(individual)
    terminals = sum(1 for node in individual if isinstance(node, gp.Terminal))
    primitives = num_nodes - terminals
    # Additional features can be added here if needed
    features = [depth, num_nodes, terminals, primitives]
    # Pad with zeros to match input_size (20)
    features += [0] * (20 - len(features))
    return features

# ===============================
# Initialize Neural Network for Fitness Prediction
# ===============================

model = FitnessPredictor(input_size=20)  # Adjust input_size if more features are added
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# ===============================
# Load and Split Training and Validation Data
# ===============================

def load_dataset(seed_file, train_size=100, val_size=10):
    """Loads the dataset from seed_file and splits into training and validation sets."""
    try:
        with open(seed_file, 'r') as f:
            seeds = json.load(f)
        seed_items = list(seeds.items())
        random.shuffle(seed_items)
        train_items = seed_items[:train_size]
        val_items = seed_items[train_size:train_size + val_size]
        return train_items, val_items
    except FileNotFoundError:
        print(f"Seed file '{seed_file}' not found.")
        return [], []

def convert_seed_to_features(seeds):
    """Convert seed data to feature and target tensors."""
    features = []
    targets = []
    for expr_id, seed_data in seeds:
        program_str = seed_data.get('program')
        input_grid = seed_data.get('input')
        output_grid = seed_data.get('output')
        if output_grid is None:
            print(f"Warning: 'output' grid is None for expr_id '{expr_id}'. Skipping.")
            continue
        fitness = compute_strict_fitness(output_grid, TARGET_GRID)
        if isinstance(program_str, str):
            try:
                individual = creator.Individual(gp.PrimitiveTree.from_string(program_str, pset))
                feature = individual_to_features(individual)
                features.append(feature)
                targets.append(fitness)
            except Exception as e:
                print(f"Failed to parse expression for ID '{expr_id}': {e}")
    if not features:
        return torch.empty(0, 20), torch.empty(0, 1)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

# ===============================
# Train the Neural Fitness Predictor Model
# ===============================

def train_predictor(model, optimizer, loss_fn, features, targets, epochs=5):
    """Train the neural network to predict fitness scores."""
    if features.size(0) == 0:
        print("No data available for training.")
        return
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

# ===============================
# Guided Selection Based on Neural Network Prediction
# ===============================

def guided_selection(population, k):
    """Select the top k individuals based on predicted fitness."""
    model.eval()
    with torch.no_grad():
        features = torch.tensor([individual_to_features(ind) for ind in population], dtype=torch.float32)
        predicted_fitnesses = model(features).squeeze().tolist()
    # Pair each individual with its predicted fitness
    individuals_with_fitness = list(zip(predicted_fitnesses, population))
    # Sort based on predicted fitness (lower is better)
    sorted_individuals = sorted(individuals_with_fitness, key=lambda x: x[0])
    # Select the top k individuals
    selected = [ind for _, ind in sorted_individuals[:k]]
    return selected

toolbox.register("select_guided", guided_selection, k=100)  # Adjust k as needed

# ===============================
# Initialize Population with Seeded Programs (Optional)
# ===============================

def initialize_population(population_size, train_seeds=None, seed_fraction=0.2):
    """Initializes a population with seeded programs and grids."""
    pop = []
    if train_seeds:
        selected_seeds = random.sample(train_seeds, min(int(population_size * seed_fraction), len(train_seeds)))
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
    # Fill the rest of the population with random individuals
    pop += [toolbox.individual() for _ in range(population_size - len(pop))]
    return pop

# ===============================
# Visualization Function
# ===============================

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
# Main Evolutionary Loop
# ===============================

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    POPULATION_SIZE = 200
    GENERATIONS = 100
    CX_PROB = 0.5  # Crossover probability
    MUT_PROB = 0.2  # Mutation probability
    SEED_FILE = 'generated_grids.json'  # Replace with your seed file if available
    TRAIN_SIZE = 100
    VAL_SIZE = 10
    
    # Load dataset
    print("Loading dataset...")
    train_seeds, val_seeds = load_dataset(SEED_FILE, train_size=TRAIN_SIZE, val_size=VAL_SIZE)
    print(f"Loaded {len(train_seeds)} training samples and {len(val_seeds)} validation samples.")
    
    # Convert training and validation seeds to tensors
    train_features, train_targets = convert_seed_to_features(train_seeds)
    val_features, val_targets = convert_seed_to_features(val_seeds)
    print(f"Training set: {train_features.shape}, Validation set: {val_features.shape}")
    
    if train_features.size(0) == 0:
        print("No valid training data available. Exiting.")
        return
    if val_features.size(0) == 0:
        print("No valid validation data available. Proceeding without validation.")
    
    # Pre-train the neural network on the training set
    print("Pre-training the neural network on training set...")
    train_predictor(model, optimizer, loss_fn, train_features, train_targets, epochs=10)
    
    if val_features.size(0) > 0:
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_features).squeeze()
            val_loss = loss_fn(val_predictions, val_targets).item()
        print(f"Validation Loss after pre-training: {val_loss:.4f}")
    
    # Initialize population with training seeds
    print("Initializing population...")
    pop = initialize_population(POPULATION_SIZE, train_seeds=train_seeds, seed_fraction=0.2)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    for gen in range(1, GENERATIONS + 1):
        print(f"=== Generation {gen} ===")
        
        # Evaluate individuals that have not yet been evaluated
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        print(f"  Evaluating {len(invalid_ind)} individuals...")
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update Hall of Fame
        hof.update(pop)

        # Gather all fitnesses for statistics
        fits = [ind.fitness.values[0] for ind in pop]

        # Record statistics
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

        # Train the neural fitness predictor model on current population
        print("  Training neural network on current population...")
        pop_features = torch.tensor([individual_to_features(ind) for ind in pop], dtype=torch.float32)
        pop_fitnesses = torch.tensor(fits, dtype=torch.float32).unsqueeze(1)
        train_predictor(model, optimizer, loss_fn, pop_features, pop_fitnesses, epochs=10)
        
        if val_features.size(0) > 0:
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_predictions = model(val_features).squeeze()
                val_loss = loss_fn(val_predictions, val_targets).item()
            print(f"  Validation Loss: {val_loss:.4f}")

        # Select the next generation population using guided selection
        print("  Selecting individuals...")
        selected = toolbox.select_guided(pop, k=POPULATION_SIZE // 2)
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
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    print("\nBest individual:", hof[0])

    # Compile and visualize the best individual
    func = toolbox.compile(expr=hof[0])
    try:
        grid_size = len(TARGET_GRID)
        generated_grid = tuple(
            tuple(func(x, y) for y in range(grid_size))
            for x in range(grid_size)
        )
        print("Generated Grid:")
        for row in generated_grid:
            print(row)
        
        print("\nTarget Grid:")
        for row in TARGET_GRID:
            print(row)
        
        visualize_grid(generated_grid, title="Best Generated Grid")
        visualize_grid(TARGET_GRID, title="Target Grid")
    except Exception as e:
        print(f"Error executing the best individual: {e}")

if __name__ == "__main__":
    main()

