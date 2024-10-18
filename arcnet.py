import os
import json
import random
import numpy as np
from termcolor import colored
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Enable Mixed Precision Training
from torch.cuda.amp import GradScaler, autocast

# Logging
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_GRID_SIZE = 30
CONTEXT_LENGTH = 8
BATCH_SIZE = 128  # Adjust based on GPU memory
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_LAYERS = 4  # Reduced number of layers
EMBED_DIM = 64
NUM_HEADS = 8
FF_DIM = 256  # Increased feedforward dimension for better capacity
PADDING_VALUE = 10

# Optional: Set PyTorch CUDA allocation configurations to optimize memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

class ARCTokenizer:
    def __init__(self, padding_value=10):
        self.vocab_size = 10**4 + 1  # 10,000 possible tokens (10x10x10x10) + 1 for padding
        self.padding_value = padding_value
        self.padding_token = self.vocab_size - 1  # Use the last token as padding token
        self.token_to_grid = {i: self._index_to_grid(i) for i in range(self.vocab_size - 1)}
        self.token_to_grid[self.padding_token] = np.full((2, 2), self.padding_value)
        self.grid_to_token = {tuple(v.flatten()): k for k, v in self.token_to_grid.items()}

    def _index_to_grid(self, index):
        return np.array([
            index // 1000,
            (index % 1000) // 100,
            (index % 100) // 10,
            index % 10
        ]).reshape(2, 2)

    def tokenize(self, grid):
        h, w = grid.shape
        tokens = []
        for i in range(0, h - 1, 2):
            for j in range(0, w - 1, 2):
                subgrid = grid[i:i+2, j:j+2]
                subgrid_tuple = tuple(subgrid.flatten())
                if all(v == self.padding_value for v in subgrid_tuple):
                    tokens.append(self.padding_token)
                else:
                    tokens.append(self.grid_to_token.get(subgrid_tuple, self.padding_token))
        return tokens

    def detokenize(self, tokens, output_shape):
        h, w = output_shape
        grid = np.full((h, w), self.padding_value, dtype=int)
        idx = 0
        for i in range(0, h - 1, 2):
            for j in range(0, w - 1, 2):
                if idx < len(tokens):
                    if tokens[idx] == self.padding_token:
                        grid[i:i+2, j:j+2] = self.padding_value
                    else:
                        grid[i:i+2, j:j+2] = self.token_to_grid[tokens[idx]]
                    idx += 1
        return grid

    def pad_grid(self, grid):
        h, w = grid.shape
        new_h = ((h + 1) // 2) * 2
        new_w = ((w + 1) // 2) * 2
        padded_grid = np.full((new_h, new_w), self.padding_value, dtype=int)
        padded_grid[:h, :w] = grid
        return padded_grid

def preprocess_grid(grid):
    """
    Converts the input grid to a list of numpy arrays.
    If the grid is a single 2D grid, it's wrapped in a list.
    If it's a sequence of grids, each is converted to a numpy array.
    """
    if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
        return [preprocess_single_grid(grid)]
    else:
        return [preprocess_single_grid(g) for g in grid]

def preprocess_single_grid(grid):
    return np.array(grid, dtype=int)

def process_colors(input_sequence, output_grid, remap_colors, replace_colors, padding_value):
    all_colors = set()
    for grid in input_sequence:
        all_colors.update(np.unique(grid))
    all_colors.update(np.unique(output_grid))
    all_colors.discard(0)  # Exclude 0 (empty cell)
    all_colors.discard(padding_value)  # Exclude padding value

    color_map = {}
    if remap_colors:
        # Remap colors to 1-9 range
        sorted_colors = sorted(all_colors)
        for i, color in enumerate(sorted_colors, start=1):
            color_map[color] = i
        if len(color_map) > 9:
            logger.warning(f"More than 9 colors found ({len(color_map)}). Some colors will be reused.")
    elif replace_colors:
        # Replace colors with random values from 1-9
        available_colors = list(range(1, 10))  # Colors 1-9
        random.shuffle(available_colors)
        for i, color in enumerate(all_colors):
            color_map[color] = available_colors[i % 9]
    else:
        # If neither remap nor replace, return original sequence and grid
        return input_sequence, output_grid

    # Ensure 0 and padding_value are not remapped
    color_map[0] = 0
    color_map[padding_value] = padding_value

    # Apply color mapping to input sequence
    processed_input = []
    for grid in input_sequence:
        processed_grid = np.vectorize(lambda x: color_map.get(x, x))(grid)
        processed_input.append(processed_grid)

    # Apply color mapping to output grid
    processed_output = np.vectorize(lambda x: color_map.get(x, x))(output_grid)

    return processed_input, processed_output

def pad_and_tokenize_sequence(sequence, tokenizer, max_grid_size, context_length, max_tokens_per_grid):
    padded_sequence = []
    for grid in sequence:
        padded_grid = tokenizer.pad_grid(grid)
        tokens = tokenizer.tokenize(padded_grid)
        # Pad or truncate tokens to max_tokens_per_grid
        tokens = tokens[:max_tokens_per_grid] + [tokenizer.padding_token] * (max_tokens_per_grid - len(tokens))
        padded_sequence.append(tokens)
    
    # Ensure we have exactly context_length frames
    if len(padded_sequence) > context_length:
        padded_sequence = padded_sequence[-context_length:]
    while len(padded_sequence) < context_length:
        padded_sequence.insert(0, [tokenizer.padding_token] * max_tokens_per_grid)
    
    return np.array(padded_sequence, dtype=np.int64)

def process_task_file(task_file, subset, padding_value, max_grid_size, context_length, max_tokens_per_grid, remap_colors, replace_colors, solutions_data=None):
    """
    Processes a single task file and returns a list of data samples.

    Args:
        task_file (str): Path to the task JSON file.
        subset (str, optional): 'train' or 'test' for evaluation JSONs. None for training JSONs.
        padding_value (int): Padding value.
        max_grid_size (int): Maximum grid size.
        context_length (int): Number of frames in the context.
        max_tokens_per_grid (int): Maximum number of tokens per grid.
        remap_colors (bool): Whether to remap colors.
        replace_colors (bool): Whether to replace colors with random new colors.
        solutions_data (dict, optional): Dictionary containing solutions for 'test' subset.

    Returns:
        list: List of tuples (input_tokens, output_tokens, output_grid_shape)
    """
    tokenizer = ARCTokenizer(padding_value=padding_value)
    data = []
    try:
        with open(task_file, 'r') as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {task_file}: {e}")
        return data

    # Determine if the JSON is a list or dict with subsets
    if subset and isinstance(task_data, dict):
        # Iterate over all top-level keys
        for sub_key, sub_data in task_data.items():
            if not isinstance(sub_data, dict):
                logger.warning(f"Sub-key '{sub_key}' in {task_file} does not contain a dictionary. Skipping.")
                continue

            examples = sub_data.get(subset, [])
            if not examples:
                logger.warning(f"No examples found for subset '{subset}' in {task_file}, sub_key '{sub_key}'. Available subsets: {list(sub_data.keys())}")
                continue

            for example in examples:
                if 'input' not in example:
                    logger.warning(f"Example missing 'input' in {task_file}, sub_key '{sub_key}'. Skipping.")
                    continue

                if subset == 'test':
                    # Get 'output' from solutions_data
                    if not solutions_data:
                        logger.error(f"solutions_data not provided for 'test' subset. Skipping.")
                        continue

                    solution_grids = solutions_data.get(sub_key, [])
                    if not solution_grids:
                        logger.warning(f"No solutions found for sub_key '{sub_key}'. Skipping.")
                        continue

                    # Assuming one 'test' example per sub_key
                    output_grid = preprocess_single_grid(solution_grids[0])
                else:
                    if 'output' not in example:
                        logger.warning(f"Example missing 'output' in {task_file}, sub_key '{sub_key}'. Skipping.")
                        continue
                    output_grid = preprocess_single_grid(example['output'])

                input_sequence = preprocess_grid(example['input'])

                if remap_colors or replace_colors:
                    input_sequence, output_grid = process_colors(
                        input_sequence, 
                        output_grid, 
                        remap_colors, 
                        replace_colors, 
                        padding_value
                    )

                # Pad and tokenize input sequence
                input_tokens = pad_and_tokenize_sequence(
                    input_sequence, 
                    tokenizer, 
                    max_grid_size, 
                    context_length, 
                    max_tokens_per_grid
                )

                # Pad and tokenize output grid
                output_padded = tokenizer.pad_grid(output_grid)
                output_tokens = tokenizer.tokenize(output_padded)
                # Pad or truncate output tokens to max_tokens_per_grid
                output_tokens = output_tokens[:max_tokens_per_grid] + [tokenizer.padding_token] * (max_tokens_per_grid - len(output_tokens))

                data.append((input_tokens, output_tokens, output_grid.shape))
    elif isinstance(task_data, list):
        # Process as training data
        for example in task_data:
            if 'input' not in example or 'output' not in example:
                logger.warning(f"Example missing 'input' or 'output' in {task_file}. Skipping.")
                continue

            input_sequence = preprocess_grid(example['input'])
            output_grid = preprocess_single_grid(example['output'])

            if remap_colors or replace_colors:
                input_sequence, output_grid = process_colors(
                    input_sequence, 
                    output_grid, 
                    remap_colors, 
                    replace_colors, 
                    padding_value
                )

            # Pad and tokenize input sequence
            input_tokens = pad_and_tokenize_sequence(
                input_sequence, 
                tokenizer, 
                max_grid_size, 
                context_length, 
                max_tokens_per_grid
            )

            # Pad and tokenize output grid
            output_padded = tokenizer.pad_grid(output_grid)
            output_tokens = tokenizer.tokenize(output_padded)
            # Pad or truncate output tokens to max_tokens_per_grid
            output_tokens = output_tokens[:max_tokens_per_grid] + [tokenizer.padding_token] * (max_tokens_per_grid - len(output_tokens))

            data.append((input_tokens, output_tokens, output_grid.shape))
    else:
        available_keys = list(task_data.keys()) if isinstance(task_data, dict) else 'N/A'
        logger.warning(f"{task_file} has an unexpected format. Available keys: {available_keys}")
        return data

    return data

class ARCDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list): List of tuples (input_tokens, output_tokens, output_grid_shape)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tokens, output_tokens, original_size = self.data[idx]
        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(output_tokens, dtype=torch.long),
            torch.tensor(original_size, dtype=torch.long)
        )

class TokenizedGridTransformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, context_length, max_tokens_per_grid, padding_value=10):
        super().__init__()
        self.tokenizer = ARCTokenizer(padding_value=padding_value)
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, embed_dim, padding_idx=self.tokenizer.padding_token)
        self.pos_encoding = nn.Embedding(context_length * max_tokens_per_grid, embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
            for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(embed_dim, self.tokenizer.vocab_size)
        self.context_length = context_length
        self.max_tokens_per_grid = max_tokens_per_grid

    def forward(self, x):
        # x shape: (batch_size, context_length, max_tokens_per_grid)
        batch_size, context_length, tokens_per_grid = x.shape
        x = x.reshape(batch_size, -1)  # Flatten the context and tokens

        # Create position indices
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.embedding(x) + self.pos_encoding(positions)

        x = x.permute(1, 0, 2)  # Change to (seq_len, batch_size, embed_dim) for transformer
        for layer in self.layers:
            x = layer(x)
        
        x = x.permute(1, 0, 2)  # Change back to (batch_size, seq_len, embed_dim)
        x = self.final_layer(x)
        
        # Reshape to (batch_size, context_length, max_tokens_per_grid, vocab_size)
        x = x.reshape(batch_size, context_length, tokens_per_grid, -1)
        
        # Only return the prediction for the last frame
        x = x[:, -1, :, :]

        return x

    def generate(self, input_sequence, max_new_tokens):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                predictions = self(input_sequence)
                next_token = predictions.argmax(dim=-1).unsqueeze(1)
                input_sequence = torch.cat([input_sequence[:, 1:], next_token], dim=1)
        return input_sequence[:, -1]

def visualize_examples(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        try:
            inputs, targets, original_sizes = next(iter(val_loader))
        except StopIteration:
            logger.warning("Validation set is empty. Skipping visualization.")
            return

        if inputs.size(0) == 0:
            logger.warning("No samples in the batch. Skipping visualization.")
            return

        idx = random.randint(0, inputs.size(0) - 1)
        input_sample = inputs[idx].unsqueeze(0).to(device)
        target_sample = targets[idx].to(device)
        
        output_sample = model(input_sample)
        
        temperature = 0.8
        output_sample = output_sample / temperature
        output_probs = torch.softmax(output_sample, dim=-1)
        predicted_sample = torch.multinomial(output_probs.view(-1, output_probs.size(-1)), num_samples=1).squeeze()
        
        unique_tokens, counts = torch.unique(predicted_sample, return_counts=True)
        logger.info(f"Unique predicted tokens: {unique_tokens.tolist()}")
        logger.info(f"Token counts: {counts.tolist()}")
        
        tokenizer = model.tokenizer
        predicted_grid = tokenizer.detokenize(predicted_sample.cpu().numpy(), (MAX_GRID_SIZE, MAX_GRID_SIZE))
        target_grid = tokenizer.detokenize(target_sample.cpu().numpy(), (MAX_GRID_SIZE, MAX_GRID_SIZE))
        
        print(colored("\nGround Truth vs Predicted", 'yellow'))
        print(f"Target shape: {target_grid.shape}, Predicted shape: {predicted_grid.shape}")
        print_grid(target_grid, predicted_grid)
        
        print("\n" + "-" * 40 + "\n")  # Separator between examples

def color_for_value(value):
    color_map = {
        0: 'white',
        1: 'red',
        2: 'green',
        3: 'yellow',
        4: 'blue',
        5: 'magenta',
        6: 'cyan',
        7: 'grey',
        8: 'light_red',
        9: 'light_green',
        10: 'light_yellow'  # Assuming 10 is your padding value
    }
    return color_map.get(value, 'white')

def print_grid(gt_grid, pred_grid):
    max_height = max(gt_grid.shape[0], pred_grid.shape[0])
    max_width = max(gt_grid.shape[1], pred_grid.shape[1])
    
    print("Ground Truth" + " " * (max_width * 3 - 5) + "Predicted")
    print("-" * (max_width * 3 - 1) + "   " + "-" * (max_width * 3 - 1))
    
    for i in range(max_height):
        gt_line = []
        pred_line = []
        for j in range(max_width):
            if i < gt_grid.shape[0] and j < gt_grid.shape[1]:
                gt_value = gt_grid[i, j]
                gt_line.append(colored(f"{gt_value:2d}", color_for_value(gt_value)))
            else:
                gt_line.append("  ")
            
            if i < pred_grid.shape[0] and j < pred_grid.shape[1]:
                pred_value = pred_grid[i, j]
                pred_line.append(colored(f"{pred_value:2d}", color_for_value(pred_value)))
            else:
                pred_line.append("  ")
        
        print(" ".join(gt_line) + "   " + " ".join(pred_line))

def exact_match_accuracy(outputs, targets, original_sizes):
    predicted = outputs.argmax(dim=-1)
    max_tokens = (MAX_GRID_SIZE // 2) ** 2
    correct = (predicted == targets).view(-1, max_tokens)  # Reshape to [batch_size, max_tokens_per_grid]
    exact_match = correct.all(dim=1).float()  # Check if all tokens in a sample match
    return exact_match.mean().item()

def cell_accuracy(outputs, targets, original_sizes):
    predicted = outputs.argmax(dim=-1)
    correct = (predicted == targets).float()
    return correct.mean().item()

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()  # For mixed precision

    best_val_loss = float('inf')
    patience = 3
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_exact_acc = 0.0
        train_cell_acc = 0.0
        for inputs, targets, original_sizes in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                # Reshape outputs and targets for loss calculation
                outputs = outputs.contiguous().view(-1, model.tokenizer.vocab_size)
                targets = targets.contiguous().view(-1)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_exact_acc += exact_match_accuracy(outputs, targets, original_sizes)
            train_cell_acc += cell_accuracy(outputs, targets, original_sizes)

        train_loss /= len(train_loader)
        train_exact_acc /= len(train_loader)
        train_cell_acc /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_exact_acc = 0.0
        val_cell_acc = 0.0
        with torch.no_grad():
            for inputs, targets, original_sizes in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast():
                    outputs = model(inputs)
                    # Reshape outputs and targets for loss calculation
                    outputs = outputs.contiguous().view(-1, model.tokenizer.vocab_size)
                    targets = targets.contiguous().view(-1)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_exact_acc += exact_match_accuracy(outputs, targets, original_sizes)
                val_cell_acc += cell_accuracy(outputs, targets, original_sizes)
        
        val_loss /= len(val_loader)
        val_exact_acc /= len(val_loader)
        val_cell_acc /= len(val_loader)
        
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Exact Acc: {train_exact_acc:.4f}, Train Cell Acc: {train_cell_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Exact Acc: {val_exact_acc:.4f}, Val Cell Acc: {val_cell_acc:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Save the best model
            torch.save(model.state_dict(), "best_tokenized_grid_transformer.pth")
            logger.info("Validation loss decreased. Saving the best model.")
        else:
            trigger_times += 1
            logger.info(f"No improvement in validation loss for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                logger.info("Early stopping triggered.")
                break

        # Visualize examples
        visualize_examples(model, val_loader, device)

        # Checkpointing every few epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"tokenized_grid_transformer_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")

def prepare_arc_data(train_tasks_dir, eval_file, solutions_file, batch_size, padding_value=10, remap_colors=False, replace_colors=False, num_workers=4, validation_subset='test'):
    """
    Prepares the training and validation DataLoaders.

    Args:
        train_tasks_dir (str): Directory containing training JSON task files.
        eval_file (str): Path to the evaluation JSON file.
        solutions_file (str): Path to the solutions JSON file.
        batch_size (int): Batch size for DataLoader.
        padding_value (int): Padding value.
        remap_colors (bool): Whether to remap colors.
        replace_colors (bool): Whether to replace colors with random new colors.
        num_workers (int): Number of parallel workers.
        validation_subset (str): Subset key to use for validation ('test', 'validation', etc.).

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Gather all training task files
    train_task_files = [
        os.path.join(train_tasks_dir, file) 
        for file in os.listdir(train_tasks_dir) 
        if file.endswith('.json')
    ]

    logger.info(f"Number of training challenge files: {len(train_task_files)}")

    if len(train_task_files) == 0:
        raise ValueError("No training challenge files found. Please check the 're_arc/tasks' directory.")

    # Process training task files in parallel
    all_train_data = []
    logger.info("Processing training task files in parallel...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_task_file, 
                task_file, 
                subset=None,  # Training files don't have subsets
                padding_value=padding_value, 
                max_grid_size=MAX_GRID_SIZE, 
                context_length=CONTEXT_LENGTH, 
                max_tokens_per_grid=(MAX_GRID_SIZE // 2) ** 2, 
                remap_colors=remap_colors, 
                replace_colors=replace_colors,
                solutions_data=None  # No solutions for training data
            )
            for task_file in train_task_files
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing training task files"):
            result = future.result()
            all_train_data.extend(result)

    logger.info(f"Total training samples loaded: {len(all_train_data)}")

    # Initialize training dataset
    train_dataset = ARCDataset(data=all_train_data)

    # Load validation data
    if not os.path.exists(eval_file):
        raise ValueError(f"Validation challenge file {eval_file} does not exist.")

    if not os.path.exists(solutions_file):
        raise ValueError(f"Solutions file {solutions_file} does not exist.")

    logger.info("Loading solutions data...")
    with open(solutions_file, 'r') as f:
        solutions_data = json.load(f)

    # Process evaluation file in parallel
    all_val_data = []
    logger.info("Processing validation task files in parallel...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_task_file, 
                task_file, 
                subset=validation_subset,  # Use specified subset for validation
                padding_value=padding_value, 
                max_grid_size=MAX_GRID_SIZE, 
                context_length=CONTEXT_LENGTH, 
                max_tokens_per_grid=(MAX_GRID_SIZE // 2) ** 2, 
                remap_colors=remap_colors, 
                replace_colors=replace_colors,
                solutions_data=solutions_data  # Pass solutions data
            )
            for task_file in [eval_file]
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing validation task files"):
            result = future.result()
            all_val_data.extend(result)

    logger.info(f"Total validation samples loaded: {len(all_val_data)}")

    # Initialize validation dataset
    val_dataset = ARCDataset(data=all_val_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # Ensure batch_size does not exceed the validation dataset size
    val_batch_size = min(batch_size, len(val_dataset)) if len(val_dataset) > 0 else 1
    val_loader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader

def main(remap_colors=False, replace_colors=False):
    logger.info(f"remap_colors: {remap_colors} replace_colors: {replace_colors}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    max_tokens_per_grid = (MAX_GRID_SIZE // 2) ** 2

    model = TokenizedGridTransformer(
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        context_length=CONTEXT_LENGTH,
        max_tokens_per_grid=max_tokens_per_grid,
        padding_value=PADDING_VALUE
    ).to(device)

    if os.path.exists("tokenized_grid_transformer_finetuned.pth"):
        logger.info("Loading pre-trained model...")
        model.load_state_dict(torch.load("tokenized_grid_transformer_finetuned.pth"))
    else:
        logger.info("Pre-trained model not found. Training from scratch...")

    train_tasks_dir = "re_arc/tasks"
    eval_file = "data/arc-agi_evaluation_challenges.json"  # Ensure this path is correct
    solutions_file = "data/arc-agi_evaluation_solutions.json"  # Path to the solutions JSON

    # Specify which subset to use for validation
    validation_subset = 'test'  # Change this if your JSON uses a different key, e.g., 'validation'

    # Prepare data loaders
    train_loader, val_loader = prepare_arc_data(
        train_tasks_dir=train_tasks_dir,
        eval_file=eval_file,
        solutions_file=solutions_file,
        batch_size=BATCH_SIZE,
        padding_value=PADDING_VALUE,
        remap_colors=remap_colors,
        replace_colors=replace_colors,
        num_workers=4,  # Adjust based on your CPU cores
        validation_subset=validation_subset
    )

    # Verify that datasets are not empty
    if len(train_loader.dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the training JSON files.")
    if len(val_loader.dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the evaluation JSON structure and subset.")

    # Optional: Inspect a few training samples
    logger.info("\nInspecting a few training samples:")
    for i in range(min(3, len(train_loader.dataset))):
        input_tokens, output_tokens, original_size = train_loader.dataset[i]
        logger.info(f"Sample {i+1}:")
        logger.info(f"Input Tokens Shape: {input_tokens.shape}")
        logger.info(f"Output Tokens Shape: {output_tokens.shape}")
        logger.info(f"Original Output Grid Size: {original_size}\n")

    # Train the model
    train_model(model, train_loader, val_loader, NUM_EPOCHS, device)

    # Save the final model
    torch.save(model.state_dict(), "tokenized_grid_transformer_finetuned.pth")
    logger.info("Final model saved as tokenized_grid_transformer_finetuned.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the TokenizedGridTransformer model")
    parser.add_argument("--remap_colors", action="store_true", help="Remap colors to a canonical list")
    parser.add_argument("--replace_colors", action="store_true", help="Replace colors with random new colors")
    parser.add_argument("--validation_subset", type=str, default='test', help="Subset key to use for validation (e.g., 'test', 'validation')")
    args = parser.parse_args()

    main(remap_colors=args.remap_colors, replace_colors=args.replace_colors)

