from termcolor import colored
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
MAX_GRID_SIZE = 30
CONTEXT_LENGTH = 8
BATCH_SIZE = 100
NUM_EPOCHS = 200
LEARNING_RATE = 5e-4
NUM_LAYERS = 10
EMBED_DIM = 64
NUM_HEADS = 8
FF_DIM = 64
PADDING_VALUE = 10


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


def process_task_file(task_file, padding_value, max_grid_size, context_length, max_tokens_per_grid, remap_colors, replace_colors):
    tokenizer = ARCTokenizer(padding_value=padding_value)
    data = []
    try:
        with open(task_file, 'r') as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {task_file}: {e}")
        return data

    if not isinstance(task_data, list):
        print(f"Warning: {task_file} does not contain a list of examples.")
        return data

    for example in task_data:
        if 'input' not in example or 'output' not in example:
            print(f"Warning: Example missing 'input' or 'output' in {task_file}. Skipping.")
            continue

        input_sequence = preprocess_grid(example['input'])
        output_grid = preprocess_single_grid(example['output'])

        if remap_colors or replace_colors:
            input_sequence, output_grid = process_colors(input_sequence, output_grid, remap_colors, replace_colors)

        # Pad and tokenize input sequence
        input_tokens = pad_and_tokenize_sequence(input_sequence, tokenizer, max_grid_size, context_length, max_tokens_per_grid)

        # Pad and tokenize output grid
        output_padded = tokenizer.pad_grid(output_grid)
        output_tokens = tokenizer.tokenize(output_padded)
        # Pad or truncate output tokens to max_tokens_per_grid
        output_tokens = output_tokens[:max_tokens_per_grid] + [tokenizer.padding_token] * (max_tokens_per_grid - len(output_tokens))

        data.append((input_tokens, output_tokens, output_grid.shape))
    return data


def preprocess_grid(grid):
    if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
        return [preprocess_single_grid(grid)]
    else:
        return [preprocess_single_grid(g) for g in grid]


def preprocess_single_grid(grid):
    return np.array(grid, dtype=int)


def process_colors(input_sequence, output_grid, remap_colors, replace_colors):
    all_colors = set()
    for grid in input_sequence:
        all_colors.update(np.unique(grid))
    all_colors.update(np.unique(output_grid))
    all_colors.discard(0)  # Exclude 0 (empty cell)
    all_colors.discard(PADDING_VALUE)  # Exclude padding value

    color_map = {}
    if remap_colors:
        # Remap colors to 1-9 range
        sorted_colors = sorted(all_colors)
        for i, color in enumerate(sorted_colors, start=1):
            color_map[color] = i
        if len(color_map) > 9:
            print(f"Warning: More than 9 colors found ({len(color_map)}). Some colors will be reused.")
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
    color_map[PADDING_VALUE] = PADDING_VALUE

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


class ARCDataset(Dataset):
    def __init__(self, task_files, max_grid_size=30, context_length=8, 
                 padding_value=10, remap_colors=False, replace_colors=False):
        self.data = []
        self.max_grid_size = max_grid_size
        self.context_length = context_length
        self.padding_value = padding_value
        self.remap_colors = remap_colors
        self.replace_colors = replace_colors

        self.tokenizer = ARCTokenizer(padding_value=padding_value)
        self.max_output_size = self.calculate_max_output_size(task_files)
        self.max_tokens_per_grid = (self.max_grid_size // 2) ** 2

        print("Loading and processing task files...")
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_task_file, 
                    task_file, 
                    padding_value, 
                    max_grid_size, 
                    context_length, 
                    self.max_tokens_per_grid, 
                    remap_colors, 
                    replace_colors
                )
                for task_file in task_files
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
                self.data.extend(future.result())

    def calculate_max_output_size(self, task_files):
        max_h, max_w = 0, 0
        for task_file in task_files:
            try:
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
            except json.JSONDecodeError:
                continue
            for example in task_data:
                if 'output' in example:
                    output_grid = example['output']
                    max_h = max(max_h, len(output_grid))
                    if len(output_grid) > 0:
                        max_w = max(max_w, len(output_grid[0]))
        return max(max_h, max_w)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tokens, output_tokens, original_size = self.data[idx]
        return (torch.tensor(input_tokens, dtype=torch.long),
                torch.tensor(output_tokens, dtype=torch.long),
                torch.tensor(original_size, dtype=torch.long))


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
            print("Validation set is empty. Skipping visualization.")
            return

        if inputs.size(0) == 0:
            print("No samples in the batch. Skipping visualization.")
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
        print("Unique predicted tokens:", unique_tokens.tolist())
        print("Token counts:", counts.tolist())
        
        predicted_grid = model.tokenizer.detokenize(predicted_sample.cpu().numpy(), (MAX_GRID_SIZE, MAX_GRID_SIZE))
        target_grid = model.tokenizer.detokenize(target_sample.cpu().numpy(), (MAX_GRID_SIZE, MAX_GRID_SIZE))
        
        print(colored("\nGround Truth vs Predicted", 'yellow'))
        print(f"Target shape: {target_grid.shape}, Predicted shape: {predicted_grid.shape}")
        print_grid(target_grid, predicted_grid)
        
        print("\n" + "-" * 40 + "\n")  # Separator between examples


def ensure_2d_grid(grid):
    if grid.ndim == 1:
        side_length = int(np.sqrt(grid.shape[0]))
        return grid.reshape(side_length, side_length)
    elif grid.ndim > 2:
        return grid.reshape(grid.shape[0], -1)
    return grid


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
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_exact_acc = 0.0
        train_cell_acc = 0.0
        for inputs, targets, original_sizes in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Reshape outputs for loss calculation
            outputs = outputs.contiguous().reshape(-1, model.tokenizer.vocab_size)
            targets = targets.reshape(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
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
            for inputs, targets, original_sizes in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Reshape outputs for loss calculation
                outputs = outputs.contiguous().reshape(-1, model.tokenizer.vocab_size)
                targets = targets.reshape(-1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_exact_acc += exact_match_accuracy(outputs, targets, original_sizes)
                val_cell_acc += cell_accuracy(outputs, targets, original_sizes)
        
        val_loss /= len(val_loader)
        val_exact_acc /= len(val_loader)
        val_cell_acc /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Exact Acc: {train_exact_acc:.4f}, Train Cell Acc: {train_cell_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Exact Acc: {val_exact_acc:.4f}, Val Cell Acc: {val_cell_acc:.4f}")

        visualize_examples(model, val_loader, device)

        # You may want to add model saving logic here
        # torch.save(model.state_dict(), f"tokenized_grid_transformer_epoch_{epoch+1}.pth")


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


def main(remap_colors=False, replace_colors=False):
    print(f"remap_colors: {remap_colors} replace_colors: {replace_colors}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load("tokenized_grid_transformer_finetuned.pth"))
    else:
        print("Pre-trained model not found. Training from scratch...")

    train_tasks_dir = "re_arc/tasks"
    eval_file = "data/arc-agi_evaluation_challenges.json"

    # Gather all training task files
    train_task_files = [os.path.join(train_tasks_dir, file) for file in os.listdir(train_tasks_dir) if file.endswith('.json')]

    print(f"Number of training challenge files: {len(train_task_files)}")

    # Initialize training dataset
    train_dataset = ARCDataset(
        task_files=train_task_files,
        max_grid_size=MAX_GRID_SIZE,
        context_length=CONTEXT_LENGTH,
        padding_value=PADDING_VALUE,
        remap_colors=remap_colors,
        replace_colors=replace_colors
    )

    total_train_samples = len(train_dataset)
    print(f"Total training samples loaded from tasks directory: {total_train_samples}")

    # Initialize validation dataset
    if not os.path.exists(eval_file):
        raise ValueError(f"Validation challenge file {eval_file} does not exist.")

    val_dataset = ARCDataset(
        task_files=[eval_file],
        max_grid_size=MAX_GRID_SIZE,
        context_length=CONTEXT_LENGTH,
        padding_value=PADDING_VALUE,
        remap_colors=remap_colors,
        replace_colors=replace_colors
    )

    print(f"Validation samples loaded from {eval_file}: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=min(BATCH_SIZE, len(val_dataset)), shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_model(model, train_loader, val_loader, NUM_EPOCHS, device)

    torch.save(model.state_dict(), "tokenized_grid_transformer_finetuned.pth")
    print("Final model saved as tokenized_grid_transformer_finetuned.pth")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the TokenizedGridTransformer model")
    parser.add_argument("--remap_colors", action="store_true", help="Remap colors to a canonical list")
    parser.add_argument("--replace_colors", action="store_true", help="Replace colors with random new colors")
    args = parser.parse_args()

    main(remap_colors=args.remap_colors, replace_colors=args.replace_colors)

