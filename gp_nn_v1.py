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
        # If generated_grid structure doesn't match target_grid
        return float('inf')

def eval_individual(individual, training_samples):
    """Evaluates an individual based on multiple training samples."""
    total_fitness = 0.0
    samples_evaluated = 0
    try:
        func = toolbox.compile(expr=individual)
        for sample_idx, sample in enumerate(training_samples):
            input_grid = sample['input']
            target_grid = sample['output']
            grid_size = len(input_grid)
            # Ensure target_grid has the same dimensions
            if len(target_grid) != grid_size or any(len(row) != len(input_grid[0]) for row in target_grid):
                print(f"Warning: Target grid size mismatch in sample {sample_idx}. Skipping this sample.")
                continue
            generated_grid = []
            for x in range(grid_size):
                generated_row = []
                for y in range(len(input_grid[0])):
                    # Pass x, y, and the entire input_grid to the program
                    cell_value = func(x, y, input_grid)
                    generated_row.append(cell_value)
                generated_grid.append(generated_row)
            fitness = compute_strict_fitness(generated_grid, target_grid)
            total_fitness += fitness
            samples_evaluated += 1
    except Exception as e:
        # Assign a high fitness value if execution fails
        print(f"Error evaluating individual: {e}")
        return (float('inf'),)
    if samples_evaluated == 0:
        # If no samples were evaluated, assign high fitness
        return (float('inf'),)
    return (total_fitness,)

toolbox.register("evaluate", eval_individual, training_samples=[])  # Placeholder, will set later
toolbox.register("mate", gp.cxOnePoint)

# ===============================
# Mutation Function Registration
# ===============================

# Use functools.partial to pass fixed parameters to gp.genFull
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
    # Frequency of each primitive
    primitive_names = [prim.name for prim in pset.primitives if hasattr(prim, 'name')]
    primitive_counts = {name: 0 for name in primitive_names}
    for node in individual:
        if isinstance(node, gp.Primitive):
            primitive_counts[node.name] += 1
    # Flatten the primitive counts into the feature vector
    primitive_features = [primitive_counts[name] for name in primitive_names]
    features = [depth, num_nodes, terminals, primitives] + primitive_features
    # Pad with zeros to match input_size (20)
    if len(features) < 20:
        features += [0] * (20 - len(features))
    else:
        features = features[:20]
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

def load_dataset(seed_file, train_size=80, val_size=20):
    """Loads the dataset from seed_file and splits into training and validation sets."""
    try:
        with open(seed_file, 'r') as f:
            seeds = json.load(f)
        seed_items = list(seeds.values())
        # Flatten all 'train' samples across all expr_ids
        all_train_samples = []
        for expr_id, data in seeds.items():
            train_samples = data.get('train', [])
            all_train_samples.extend(train_samples)
        # Shuffle the training samples
        random.shuffle(all_train_samples)
        # Filter out samples with size mismatches
        valid_train_samples = []
        for idx, sample in enumerate(all_train_samples):
            input_grid = sample.get('input')
            output_grid = sample.get('output')
            if input_grid and output_grid:
                if len(input_grid) == len(output_grid) and all(len(row_input) == len(row_output) for row_input, row_output in zip(input_grid, output_grid)):
                    valid_train_samples.append(sample)
                else:
                    print(f"Warning: Target grid size mismatch in sample {idx}. Skipping this sample.")
            else:
                print(f"Warning: Missing 'input' or 'output' in sample {idx}. Skipping this sample.")
            if len(valid_train_samples) >= train_size + val_size:
                break
        # Shuffle the valid samples again to ensure randomness
        random.shuffle(valid_train_samples)
        train_samples = valid_train_samples[:train_size]
        val_samples = valid_train_samples[train_size:train_size + val_size]
        return train_samples, val_samples
    except FileNotFoundError:
        print(f"Seed file '{seed_file}' not found.")
        return [], []

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

def guided_selection(population, k, model):
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

# Register the guided_selection with partial to include the model
def select_guided(population, k):
    return guided_selection(population, k, model)

toolbox.register("select_guided", select_guided, k=80)  # 80 for training

# ===============================
# Initialize Population with Seeded Programs (Optional)
# ===============================

def initialize_population(population_size, training_samples=None, seed_fraction=0.2):
    """Initializes a population with seeded programs and grids."""
    pop = []
    if training_samples:
        selected_seeds = random.sample(training_samples, min(int(population_size * seed_fraction), len(training_samples)))
        for sample_idx, seed_data in enumerate(selected_seeds):
            # Since we don't have program strings, we'll initialize randomly
            individual = toolbox.individual()
            pop.append(individual)
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
    SEED_FILE = 'arc-agi_training_challenges.json'  # Use original dataset
    TRAIN_SIZE = 80
    VAL_SIZE = 20
    
    # Load dataset
    print("Loading dataset...")
    train_seeds, val_seeds = load_dataset(SEED_FILE, train_size=TRAIN_SIZE, val_size=VAL_SIZE)
    print(f"Loaded {len(train_seeds)} training samples and {len(val_seeds)} validation samples.")
    
    if len(train_seeds) < TRAIN_SIZE:
        print(f"Insufficient training samples. Required: {TRAIN_SIZE}, Available: {len(train_seeds)}")
    if len(val_seeds) < VAL_SIZE:
        print(f"Insufficient validation samples. Required: {VAL_SIZE}, Available: {len(val_seeds)}")
    
    # Initialize population
    print("Initializing population...")
    pop = initialize_population(POPULATION_SIZE, training_samples=train_seeds, seed_fraction=0.2)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    # Initialize training data for the neural network
    nn_training_features = []
    nn_training_fitnesses = []

    for gen in range(1, GENERATIONS + 1):
        print(f"=== Generation {gen} ===")
        
        # Evaluate individuals that have not yet been evaluated
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        print(f"  Evaluating {len(invalid_ind)} individuals...")
        fitnesses = [eval_individual(ind, train_seeds)[0] for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
            # Collect features and fitnesses for neural network training
            nn_training_features.append(individual_to_features(ind))
            nn_training_fitnesses.append(fit)
        
        # Update Hall of Fame
        hof.update(pop)

        # Gather all fitnesses for statistics
        fits = [ind.fitness.values[0] for ind in pop]

        # Record statistics
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

        # Train the neural fitness predictor model on collected data
        if len(nn_training_features) > 0:
            print("  Training neural network on collected data...")
            features_tensor = torch.tensor(nn_training_features, dtype=torch.float32)
            fitness_tensor = torch.tensor(nn_training_fitnesses, dtype=torch.float32).unsqueeze(1)
            train_predictor(model, optimizer, loss_fn, features_tensor, fitness_tensor, epochs=10)
        else:
            print("  No new data to train the neural network.")
        
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
    fitnesses = [eval_individual(ind, train_seeds)[0] for ind in invalid_ind]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = (fit,)
        # Optionally, collect more data for the neural network
        nn_training_features.append(individual_to_features(ind))
        nn_training_fitnesses.append(fit)

    hof.update(pop)

    print("\nBest individual:", hof[0])

    # Compile and visualize the best individual
    func = toolbox.compile(expr=hof[0])
    try:
        # For visualization, select a specific validation sample
        if len(val_seeds) > 0:
            sample = val_seeds[0]  # You can choose different samples
            input_grid = sample['input']
            target_grid = sample['output']
            grid_size = len(input_grid)
            generated_grid = []
            for x in range(grid_size):
                generated_row = []
                for y in range(len(input_grid[0])):
                    cell_value = func(x, y, input_grid)
                    generated_row.append(cell_value)
                generated_grid.append(generated_row)
            print("Generated Grid:")
            for row in generated_grid:
                print(row)
            
            print("\nTarget Grid:")
            for row in target_grid:
                print(row)
            
            visualize_grid(generated_grid, title="Best Generated Grid")
            visualize_grid(target_grid, title="Target Grid")
        else:
            print("No validation samples available for visualization.")
    except Exception as e:
        print(f"Error executing the best individual: {e}")

