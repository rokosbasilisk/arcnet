import json
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from itertools import product  # For generating hyperparameter combinations

# Constants (Adjusted)
MAX_GRID_SIZE = 30
CONTEXT_LENGTH = 9
PADDING_VALUE = 10
NUM_EPOCHS = 100

class ARCTokenizer:
    def __init__(self, padding_value=10):
        self.vocab_size = 10**4 + 1  # 10,000 possible tokens + 1 for padding
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

class ARCDataset(Dataset):
    def __init__(self, challenges_file, solutions_file, max_grid_size=30, context_length=6, 
                 padding_value=10, remap_colors=False, replace_colors=False):
        with open(challenges_file, 'r') as f:
            self.challenges = json.load(f)
        with open(solutions_file, 'r') as f:
            self.solutions = json.load(f)
        
        self.data = []
        self.max_grid_size = max_grid_size
        self.context_length = context_length
        self.padding_value = padding_value
        self.remap_colors = remap_colors
        self.replace_colors = replace_colors
        
        self.tokenizer = ARCTokenizer(padding_value=padding_value)
        self.max_output_size = self.calculate_max_output_size()
        self.max_tokens_per_grid = (self.max_grid_size // 2) ** 2
        
        self.process_data()

    def calculate_max_output_size(self):
        max_h, max_w = 0, 0
        for challenge_id in self.challenges:
            for train_pair in self.challenges[challenge_id]['train']:
                output_grid = train_pair['output']
                max_h = max(max_h, len(output_grid))
                max_w = max(max_w, len(output_grid[0]))
        return max(max_h, max_w)

    def preprocess_grid(self, grid):
        if isinstance(grid[0], list):  # If it's already a 2D grid
            return [self.preprocess_single_grid(grid)]
        else:  # If it's a sequence of grids
            return [self.preprocess_single_grid(g) for g in grid]

    def preprocess_single_grid(self, grid):
        return np.array(grid, dtype=int)

    def process_colors(self, input_sequence, output_grid):
        all_colors = set()
        for grid in input_sequence:
            all_colors.update(np.unique(grid))
        all_colors.update(np.unique(output_grid))
        all_colors.discard(0)  # Exclude 0 (empty cell)
        all_colors.discard(self.padding_value)  # Exclude padding value

        if self.remap_colors:
            # Remap colors to 1-9 range
            color_map = {color: i for i, color in enumerate(sorted(all_colors), start=1)}
            if len(color_map) > 9:
                print(f"Warning: More than 9 colors found ({len(color_map)}). Some colors will be reused.")
        elif self.replace_colors:
            # Replace colors with random values from 1-9
            available_colors = list(range(1, 10))  # Colors 1-9
            random.shuffle(available_colors)
            color_map = {color: available_colors[i % 9] for i, color in enumerate(all_colors)}
        else:
            # If neither remap nor replace, return original sequence and grid
            return input_sequence, output_grid

        # Ensure 0 and padding_value are not remapped
        color_map[0] = 0
        color_map[self.padding_value] = self.padding_value

        # Apply color mapping to input sequence
        processed_input = []
        for grid in input_sequence:
            processed_grid = np.vectorize(lambda x: color_map.get(x, x))(grid)
            processed_input.append(processed_grid)

        # Apply color mapping to output grid
        processed_output = np.vectorize(lambda x: color_map.get(x, x))(output_grid)

        return processed_input, processed_output

    def pad_and_tokenize_sequence(self, sequence):
        padded_sequence = []
        for grid in sequence:
            padded_grid = self.tokenizer.pad_grid(grid)
            tokens = self.tokenizer.tokenize(padded_grid)
            # Pad or truncate tokens to max_tokens_per_grid
            tokens = tokens[:self.max_tokens_per_grid] + [self.tokenizer.padding_token] * (self.max_tokens_per_grid - len(tokens))
            padded_sequence.append(tokens)
        
        # Ensure we have exactly context_length frames
        if len(padded_sequence) > self.context_length:
            padded_sequence = padded_sequence[-self.context_length:]
        while len(padded_sequence) < self.context_length:
            padded_sequence.insert(0, [self.tokenizer.padding_token] * self.max_tokens_per_grid)
        
        return np.array(padded_sequence, dtype=np.int64)

    def augment_grid(self, grid):
        # Example augmentation: random rotation
        rotations = [np.rot90(grid, k) for k in range(4)]
        return random.choice(rotations)

    def process_data(self):
        for challenge_id in self.challenges:
            challenge = self.challenges[challenge_id]
            solution = self.solutions[challenge_id]
            
            for train_pair in challenge['train']:
                input_sequence = self.preprocess_grid(train_pair['input'])
                output_grid = self.preprocess_single_grid(train_pair['output'])
                
                if self.remap_colors or self.replace_colors:
                    input_sequence, output_grid = self.process_colors(input_sequence, output_grid)
                
                # Apply augmentation
                augmented_input_sequence = [self.augment_grid(grid) for grid in input_sequence]
                augmented_output_grid = self.augment_grid(output_grid)
                
                # Pad and tokenize input sequence
                input_tokens = self.pad_and_tokenize_sequence(augmented_input_sequence)
                
                # Pad and tokenize output grid
                output_padded = self.tokenizer.pad_grid(augmented_output_grid)
                output_tokens = self.tokenizer.tokenize(output_padded)
                # Pad or truncate output tokens to max_tokens_per_grid
                output_tokens = output_tokens[:self.max_tokens_per_grid] + [self.tokenizer.padding_token] * (self.max_tokens_per_grid - len(output_tokens))
                
                self.data.append((input_tokens, output_tokens, augmented_output_grid.shape))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tokens, output_tokens, original_size = self.data[idx]
        return (torch.tensor(input_tokens, dtype=torch.long),
                torch.tensor(output_tokens, dtype=torch.long),
                torch.tensor(original_size, dtype=torch.long))

def prepare_arc_data(train_file, eval_file, batch_size, padding_value=10, remap_colors=False, replace_colors=False):
    # Load training dataset
    train_dataset = ARCDataset(train_file, train_file.replace('challenges', 'solutions'), 
                               context_length=CONTEXT_LENGTH,
                               padding_value=padding_value, remap_colors=remap_colors, replace_colors=replace_colors)
    
    # Load evaluation dataset for validation
    val_dataset = ARCDataset(eval_file, eval_file.replace('challenges', 'solutions'), 
                             context_length=CONTEXT_LENGTH,
                             padding_value=padding_value, remap_colors=remap_colors, replace_colors=replace_colors)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


class TokenizedGridTransformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, context_length, max_tokens_per_grid, padding_value=10):
        super().__init__()
        self.tokenizer = ARCTokenizer(padding_value=padding_value)
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, embed_dim, padding_idx=self.tokenizer.padding_token)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, context_length * max_tokens_per_grid, embed_dim))
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.5)
            for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(embed_dim, self.tokenizer.vocab_size)
        self.context_length = context_length
        self.max_tokens_per_grid = max_tokens_per_grid

    def forward(self, x):
        # x shape: (batch_size, context_length, max_tokens_per_grid)
        batch_size, context_length, tokens_per_grid = x.shape
        x = x.reshape(batch_size, context_length * tokens_per_grid)  # Flatten context and tokens

        # Get token embeddings
        token_embeddings = self.embedding(x)  # (batch_size, context_length * tokens_per_grid, embed_dim)
        
        # Add positional embeddings
        embeddings = token_embeddings + self.pos_encoder  # (batch_size, context_length * tokens_per_grid, embed_dim)
        
        # Prepare for Transformer: (seq_len, batch_size, embed_dim)
        embeddings = embeddings.permute(1, 0, 2)
        
        # Pass through Transformer layers
        for layer in self.layers:
            embeddings = layer(embeddings)
        
        # Back to (batch_size, seq_len, embed_dim)
        embeddings = embeddings.permute(1, 0, 2)
        
        # Final linear layer
        outputs = self.final_layer(embeddings)  # (batch_size, seq_len, vocab_size)
        
        # Reshape to (batch_size, context_length, max_tokens_per_grid, vocab_size)
        outputs = outputs.reshape(batch_size, self.context_length, self.max_tokens_per_grid, -1)
        
        # Only return the prediction for the last frame
        outputs = outputs[:, -1, :, :]  # (batch_size, max_tokens_per_grid, vocab_size)
        
        return outputs

    def generate(self, input_sequence, max_new_tokens):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                predictions = self(input_sequence)
                next_token = predictions.argmax(dim=-1).unsqueeze(1)
                input_sequence = torch.cat([input_sequence[:, 1:], next_token], dim=1)
        return input_sequence[:, -1]

def plot_train_val_curves(train_losses, val_losses):
    """Plot the training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title('Training and validation loss curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_model(model, train_loader, val_loader, num_epochs, device, is_vis, hyperparams):
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['LEARNING_RATE'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait before early stopping
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets, original_sizes in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Reshape outputs for loss calculation
            outputs_reshaped = outputs.contiguous().reshape(-1, model.tokenizer.vocab_size)
            targets_reshaped = targets.reshape(-1)
            
            loss = criterion(outputs_reshaped, targets_reshaped)
            loss.backward()
            
            # Apply gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, original_sizes in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Reshape outputs for loss calculation
                outputs_reshaped = outputs.contiguous().reshape(-1, model.tokenizer.vocab_size)
                targets_reshaped = targets.reshape(-1)
                
                loss = criterion(outputs_reshaped, targets_reshaped)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Overfit: {(val_loss-train_loss):.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Plot loss curves
    if is_vis:
        plot_train_val_curves(train_losses, val_losses)
    
    return best_val_loss

def main(remap_colors=False, replace_colors=False, is_vis=False):
    print(f"remap_colors: {remap_colors} replace_colors: {replace_colors}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    max_tokens_per_grid = (MAX_GRID_SIZE // 2) ** 2

    # Define hyperparameter ranges
    num_layers_list = [3, 5, 7]
    embed_dim_list = [32, 64]
    ff_dim_list = [64, 128]
    num_heads_list = [2, 4]
    learning_rate_list = [1e-4, 3e-4]
    batch_size_list = [16, 32]

    # Generate all combinations of hyperparameters
    hyperparams_combinations = list(product(num_layers_list, embed_dim_list, ff_dim_list, num_heads_list, learning_rate_list, batch_size_list))

    best_val_loss = float('inf')
    best_hyperparams = None

    for hyperparams_values in hyperparams_combinations:
        hyperparams = {
            'NUM_LAYERS': hyperparams_values[0],
            'EMBED_DIM': hyperparams_values[1],
            'FF_DIM': hyperparams_values[2],
            'NUM_HEADS': hyperparams_values[3],
            'LEARNING_RATE': hyperparams_values[4],
            'BATCH_SIZE': hyperparams_values[5]
        }
        print(f"\nTraining with hyperparameters: {hyperparams}")

        model = TokenizedGridTransformer(
            num_layers=hyperparams['NUM_LAYERS'],
            embed_dim=hyperparams['EMBED_DIM'],
            num_heads=hyperparams['NUM_HEADS'],
            ff_dim=hyperparams['FF_DIM'],
            context_length=CONTEXT_LENGTH,
            max_tokens_per_grid=max_tokens_per_grid,
            padding_value=PADDING_VALUE
        ).to(device)

        train_loader, val_loader = prepare_arc_data(
            "data/arc-agi_training_challenges.json",
            "data/arc-agi_evaluation_challenges.json",
            batch_size=hyperparams['BATCH_SIZE'],
            padding_value=PADDING_VALUE,
            remap_colors=remap_colors,
            replace_colors=replace_colors
        )

        val_loss = train_model(model, train_loader, val_loader, NUM_EPOCHS, device, is_vis, hyperparams)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hyperparams = hyperparams
            # Save the best model
            torch.save(model.state_dict(), "best_model.pth")

    print(f"\nBest validation loss: {best_val_loss:.4f} with hyperparameters: {best_hyperparams}")
    print("Best model saved as best_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TokenizedGridTransformer model with hyperparameter sweep")
    parser.add_argument("--remap_colors", action="store_true", help="Remap colors to a canonical list")
    parser.add_argument("--replace_colors", action="store_true", help="Replace colors with random new colors")
    parser.add_argument("--is_vis", action="store_true", help="Enable visualization after training")
    args = parser.parse_args()

    main(remap_colors=args.remap_colors, replace_colors=args.replace_colors, is_vis=args.is_vis)


