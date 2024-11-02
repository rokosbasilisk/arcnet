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
# Siamese Network Model to Predict Fitness
# ===============================

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Feature extractor for each grid using Convolutional Layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [1, 30, 30] -> [16, 30, 30]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [16, 15, 15]
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # [32, 15, 15]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [32, 7, 7]
            nn.Flatten(),                                 # [32*7*7 = 1568]
            nn.Linear(1568, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # Distance computation
        self.distance_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1)
        f2 = self.feature_extractor(x2)
        # Compute absolute difference
        diff = torch.abs(f1 - f2)
        distance = self.distance_layer(diff)
        return distance

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

def compute_strict_fitness(generated_grid, target_grid):
    """Calculates a strict fitness score with MSE and penalty for mismatches."""
    if generated_grid is None:
        return float('inf')
    try:
        mse = sum((cell_gen - cell_tar) ** 2 for row_gen, row_tar in zip(generated_grid, target_grid)
                  for cell_gen, cell_tar in zip(row_gen, row_tar))
        penalty = sum(1 for row_gen, row_tar in zip(generated_grid, target_grid)
                      for cell_gen, cell_tar in zip(row_gen, row_tar) if cell_gen != cell_tar)
        # Normalize fitness by the number of cells to keep it manageable
        total_cells = len(target_grid) * len(target_grid[0])
        return (mse + penalty) / total_cells
    except TypeError:
        return float('inf')

def eval_individual(individual, training_samples, network, device):
    """Evaluates an individual based on multiple training samples using the Siamese network."""
    total_fitness = 0.0
    samples_evaluated = 0
    try:
        func = toolbox.compile(expr=individual)
        network.eval()
        with torch.no_grad():
            for idx, sample in enumerate(training_samples):
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
                # Convert grids to tensors and normalize
                generated_tensor = torch.tensor(generated_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, 30, 30]
                target_tensor = torch.tensor(target_grid_padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)    # Shape: [1, 1, 30, 30]
                # Normalize tensors to [0, 1]
                generated_tensor /= 9.0
                target_tensor /= 9.0
                # Compute distance using the network
                distance = network(generated_tensor, target_tensor)
                if distance is not None and distance.numel() > 0:
                    fitness = distance.item()
                else:
                    print(f"Warning: Distance is invalid for sample {idx}. Assigning high fitness.")
                    fitness = float('inf')
                total_fitness += fitness
                samples_evaluated += 1
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        return (float('inf'),)
    if samples_evaluated == 0:
        return (float('inf'),)
    # Return average fitness over samples
    return (total_fitness / samples_evaluated,)

toolbox.register("evaluate", eval_individual, training_samples=[], network=None, device='cpu')  # Placeholder, will set later
toolbox.register("mate", gp.cxOnePoint)

# ===============================
# Mutation Function Registration
# ===============================

mutate_expr = partial(gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=mutate_expr, pset=pset)

# ===============================
# Siamese Network Training Functions
# ===============================

def prepare_dataset(seeds):
    """Prepares grid pairs and their distances from seed data."""
    pairs = []
    distances = []
    for data in seeds.values():
        train_samples = data.get('train', [])
        for sample in train_samples:
            input_grid = sample.get('input')
            target_grid = sample.get('output')
            if input_grid and target_grid:
                input_padded = pad_grid(input_grid, 30, 30)
                target_padded = pad_grid(target_grid, 30, 30)
                # Compute distance (MSE)
                mse = np.mean((np.array(input_padded) - np.array(target_padded)) ** 2)
                pairs.append((input_padded, target_padded))
                distances.append(mse)
    return pairs, distances

def load_dataset_for_network(seed_file, train_size=80, val_size=20):
    """Loads the dataset from seed_file and splits into training and validation sets for the network."""
    try:
        with open(seed_file, 'r') as f:
            seeds = json.load(f)
        pairs, distances = prepare_dataset(seeds)
        # Shuffle the data
        combined = list(zip(pairs, distances))
        random.shuffle(combined)
        if combined:
            pairs[:], distances[:] = zip(*combined)
        else:
            pairs, distances = [], []
        # Split into training and validation
        train_pairs = pairs[:train_size]
        train_distances = distances[:train_size]
        val_pairs = pairs[train_size:train_size + val_size]
        val_distances = distances[train_size:train_size + val_size]
        return (train_pairs, train_distances), (val_pairs, val_distances)
    except FileNotFoundError:
        print(f"Seed file '{seed_file}' not found.")
        return ([], []), ([], [])

def train_siamese_network(model, optimizer, loss_fn, train_pairs, train_distances, val_pairs, val_distances, device, epochs=50, batch_size=16):
    """Trains the Siamese network with batch processing and validation."""
    model.to(device)
    # Prepare training data
    train_inputs = torch.tensor([pair[0] for pair in train_pairs], dtype=torch.float32).unsqueeze(1)  # Shape: [N, 1, 30, 30]
    train_targets = torch.tensor([pair[1] for pair in train_pairs], dtype=torch.float32).unsqueeze(1)  # Shape: [N, 1, 30, 30]
    train_distances = torch.tensor(train_distances, dtype=torch.float32).unsqueeze(1)  # Shape: [N, 1]

    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets, train_distances)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare validation data
    val_inputs = torch.tensor([pair[0] for pair in val_pairs], dtype=torch.float32).unsqueeze(1).to(device)
    val_targets = torch.tensor([pair[1] for pair in val_pairs], dtype=torch.float32).unsqueeze(1).to(device)
    val_distances = torch.tensor(val_distances, dtype=torch.float32).unsqueeze(1).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_targets, batch_distances in train_loader:
            optimizer.zero_grad()
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_distances = batch_distances.to(device)
            # Normalize tensors to [0, 1]
            batch_inputs /= 9.0
            batch_targets /= 9.0
            # Compute distance
            predicted_distances = model(batch_inputs, batch_targets)
            # Compute loss
            loss = loss_fn(predicted_distances, batch_distances)
            loss.backward()
            # Gradient Clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0

        # Validation
        model.eval()
        with torch.no_grad():
            val_inputs_normalized = val_inputs / 9.0
            val_targets_normalized = val_targets / 9.0
            predicted_val_distances = model(val_inputs_normalized, val_targets_normalized)
            val_loss = loss_fn(predicted_val_distances, val_distances).item()

        print(f"Epoch {epoch}/{epochs} - Training Loss: {avg_train_loss:.4f} - Validation Loss: {val_loss:.4f}")

    print("Training completed.")

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

    # Initialize Siamese Network
    network = SiameseNetwork()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)  # Lower learning rate
    loss_fn = nn.MSELoss()

    # Load and prepare dataset for training the Siamese network
    SEED_FILE = 'data/arc-agi_training_challenges.json'
    TRAIN_SIZE_NET = 80
    VAL_SIZE_NET = 20

    print("Loading dataset for Siamese network training...")
    (train_pairs, train_distances), (val_pairs, val_distances) = load_dataset_for_network(SEED_FILE, train_size=TRAIN_SIZE_NET, val_size=VAL_SIZE_NET)
    print(f"Prepared {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs.")

    if len(train_pairs) < TRAIN_SIZE_NET:
        print(f"Warning: Insufficient training pairs. Required: {TRAIN_SIZE_NET}, Available: {len(train_pairs)}")
    if len(val_pairs) < VAL_SIZE_NET:
        print(f"Warning: Insufficient validation pairs. Required: {VAL_SIZE_NET}, Available: {len(val_pairs)}")

    # Normalize grid values to [0, 1]
    def normalize_grid(grid):
        return [[cell / 9.0 for cell in row] for row in grid]

    train_pairs = [(normalize_grid(pair[0]), normalize_grid(pair[1])) for pair in train_pairs]
    val_pairs = [(normalize_grid(pair[0]), normalize_grid(pair[1])) for pair in val_pairs]

    # Train the Siamese network
    print("Training the Siamese network...")
    train_siamese_network(network, optimizer, loss_fn, train_pairs, train_distances, val_pairs, val_distances, device, epochs=50, batch_size=16)

    # Save the trained network
    model_path = 'siamese_network.pth'
    torch.save(network.state_dict(), model_path)
    print(f"Siamese network saved to {model_path}")

    # Initialize GP components
    print("Initializing Genetic Programming components...")
    POPULATION_SIZE = 100
    GENERATIONS = 100
    CX_PROB = 0.5  # Crossover probability
    MUT_PROB = 0.2  # Mutation probability

    # Load the dataset again for GP training
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
    training_samples = valid_train_samples[:TRAIN_SIZE_NET]
    validation_samples = valid_train_samples[TRAIN_SIZE_NET:TRAIN_SIZE_NET + VAL_SIZE_NET]
    print(f"Using {len(training_samples)} training samples for GP.")

    # Initialize population with seeded individuals
    pop = []
    seed_fraction = 0.2
    seed_count = min(int(POPULATION_SIZE * seed_fraction), len(training_samples))
    selected_seeds = random.sample(training_samples, seed_count)
    for _ in selected_seeds:
        # Initialize randomly; can be enhanced to incorporate seed data
        individual = toolbox.individual()
        pop.append(individual)
    # Fill the rest of the population with random individuals
    pop += [toolbox.individual() for _ in range(POPULATION_SIZE - len(pop))]
    print(f"Initialized population with {len(pop)} individuals.")

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    # Load the trained Siamese network
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.to(device)
    network.eval()

    # Update the toolbox evaluate function with the trained network and device
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", eval_individual, training_samples=training_samples, network=network, device=device)

    # Evolutionary Loop
    for gen in range(1, GENERATIONS + 1):
        print(f"=== Generation {gen} ===")

        # Evaluate individuals that have not yet been evaluated
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        print(f"  Evaluating {len(invalid_ind)} individuals...")
        fitnesses = []
        for ind in invalid_ind:
            fit = toolbox.evaluate(ind)
            fitnesses.append(fit)
            if fit is None:
                print("Warning: Fitness evaluation returned None.")
        for ind, fit in zip(invalid_ind, fitnesses):
            if fit is not None:
                ind.fitness.values = fit
            else:
                ind.fitness.values = (float('inf'),)

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
                # Select a sample from training or validation for visualization
                if len(validation_samples) > 0:
                    sample = validation_samples[0]  # You can choose different samples or iterate over multiple
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
        selected = tools.selTournament(pop, k=POPULATION_SIZE // 2, tournsize=3)  # Using tournament selection
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
    fitnesses = []
    for ind in invalid_ind:
        fit = toolbox.evaluate(ind)
        fitnesses.append(fit)
        if fit is None:
            print("Warning: Fitness evaluation returned None.")
    for ind, fit in zip(invalid_ind, fitnesses):
        if fit is not None:
            ind.fitness.values = fit
        else:
            ind.fitness.values = (float('inf'),)

    hof.update(pop)

    print("\nBest individual:", hof[0])

    # Compile and visualize the best individual
    func = toolbox.compile(expr=hof[0])
    try:
        if len(validation_samples) > 0:
            sample = validation_samples[0]  # You can choose different samples or iterate over multiple
            input_grid = sample['input']
            target_grid = sample['output']
            # Enforce fixed grid size of 30x30 by padding if necessary
            input_grid_padded = pad_grid(input_grid, 30, 30)
            target_grid_padded = pad_grid(target_grid, 30, 30)
            # Normalize grids
            input_grid_normalized = [[cell / 9.0 for cell in row] for row in input_grid_padded]
            target_grid_normalized = [[cell / 9.0 for cell in row] for row in target_grid_padded]
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

