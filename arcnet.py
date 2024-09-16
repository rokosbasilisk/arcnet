from termcolor import colored
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from tqdm import tqdm
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import matplotlib.pyplot as plt
from datasets import load_metric
from transformers import TrainerCallback, TrainingArguments, Trainer

# Constants
MAX_GRID_SIZE = 30
CONTEXT_LENGTH = 8
BATCH_SIZE = 100
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_LAYERS = 5
EMBED_DIM = 64  # Adjusted to accommodate multi-scale embeddings
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

    def process_data(self):
        for challenge_id in self.challenges:
            challenge = self.challenges[challenge_id]
            solution = self.solutions[challenge_id]
            
            for train_pair in challenge['train']:
                input_sequence = self.preprocess_grid(train_pair['input'])
                output_grid = self.preprocess_single_grid(train_pair['output'])
                
                if self.remap_colors or self.replace_colors:
                    input_sequence, output_grid = self.process_colors(input_sequence, output_grid)
                
                # Pad and tokenize input sequence
                input_tokens = self.pad_and_tokenize_sequence(input_sequence)
                
                # Pad and tokenize output grid
                output_padded = self.tokenizer.pad_grid(output_grid)
                output_tokens = self.tokenizer.tokenize(output_padded)
                # Pad or truncate output tokens to max_tokens_per_grid
                output_tokens = output_tokens[:self.max_tokens_per_grid] + [self.tokenizer.padding_token] * (self.max_tokens_per_grid - len(output_tokens))
                
                self.data.append((input_tokens, output_tokens, output_grid.shape))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tokens, output_tokens, original_size = self.data[idx]
        return (torch.tensor(input_tokens, dtype=torch.long),
                torch.tensor(output_tokens, dtype=torch.long),
                torch.tensor(original_size, dtype=torch.long))

def prepare_arc_data(train_file, eval_file, batch_size, padding_value=10, val_fraction=0.1, remap_colors=False, replace_colors=False):
    # Load both datasets
    full_dataset = ARCDataset(train_file, train_file.replace('challenges', 'solutions'), 
                              padding_value=padding_value, remap_colors=remap_colors, replace_colors=replace_colors)
    full_dataset.data.extend(ARCDataset(eval_file, eval_file.replace('challenges', 'solutions'), 
                                        padding_value=padding_value, remap_colors=remap_colors, replace_colors=replace_colors).data)

    # Shuffle the combined dataset
    random.shuffle(full_dataset.data)
    
    # Calculate the split
    dataset_size = len(full_dataset)
    val_size = max(int(dataset_size * val_fraction), 1)  # Ensure at least one validation sample
    train_size = dataset_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size))
    
    print(f"Full dataset size: {dataset_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

class TokenizedGridTransformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, context_length, max_tokens_per_grid, padding_value=10, max_grid_size=30):
        super().__init__()
        self.tokenizer = ARCTokenizer(padding_value=padding_value)
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, embed_dim, padding_idx=self.tokenizer.padding_token)
        
        # Multi-Scale Positional Encoding
        # Scale 1: Fine Scale (e.g., 15x15 for 30x30 grid with 2x2 tokens)
        scale1_size = max_grid_size // 2  # 15
        self.row_pos_encoding_scale1 = nn.Embedding(scale1_size, embed_dim // 2)
        self.col_pos_encoding_scale1 = nn.Embedding(scale1_size, embed_dim // 2)
        
        # Scale 2: Coarse Scale (e.g., 5x5 by grouping 3x3 tokens)
        scale2_size = scale1_size // 3  # 5
        self.row_pos_encoding_scale2 = nn.Embedding(scale2_size, embed_dim // 2)
        self.col_pos_encoding_scale2 = nn.Embedding(scale2_size, embed_dim // 2)
        
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
        x = x.reshape(batch_size, context_length * tokens_per_grid)  # Flatten context and tokens

        # Compute token grid positions
        grid_size_scale1 = self.max_tokens_per_grid // (self.max_tokens_per_grid // 15)  # 15
        grid_size_scale2 = grid_size_scale1 // 3  # 5

        # Compute row and column indices for each token at scale1
        token_idx = torch.arange(tokens_per_grid, device=x.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, tokens_per_grid)
        rows_scale1 = (token_idx // (tokens_per_grid // (self.max_tokens_per_grid // 15))) % 15  # (batch_size, tokens_per_grid)
        cols_scale1 = token_idx % 15  # (batch_size, tokens_per_grid)

        # Compute row and column indices for each token at scale2
        rows_scale2 = rows_scale1 // 3  # Integer division for coarse grouping
        cols_scale2 = cols_scale1 // 3  # Integer division for coarse grouping

        # Expand to match context_length
        rows_scale1 = rows_scale1.unsqueeze(1).repeat(1, context_length, 1).reshape(batch_size, -1)  # (batch_size, context_length * tokens_per_grid)
        cols_scale1 = cols_scale1.unsqueeze(1).repeat(1, context_length, 1).reshape(batch_size, -1)  # (batch_size, context_length * tokens_per_grid)
        
        rows_scale2 = rows_scale2.unsqueeze(1).repeat(1, context_length, 1).reshape(batch_size, -1)  # (batch_size, context_length * tokens_per_grid)
        cols_scale2 = cols_scale2.unsqueeze(1).repeat(1, context_length, 1).reshape(batch_size, -1)  # (batch_size, context_length * tokens_per_grid)

        # Get positional embeddings for both scales
        row_emb_scale1 = self.row_pos_encoding_scale1(rows_scale1)  # (batch_size, context_length * tokens_per_grid, embed_dim//2)
        col_emb_scale1 = self.col_pos_encoding_scale1(cols_scale1)  # (batch_size, context_length * tokens_per_grid, embed_dim//2)
        
        row_emb_scale2 = self.row_pos_encoding_scale2(rows_scale2)  # (batch_size, context_length * tokens_per_grid, embed_dim//2)
        col_emb_scale2 = self.col_pos_encoding_scale2(cols_scale2)  # (batch_size, context_length * tokens_per_grid, embed_dim//2)

        # Combine embeddings from both scales
        pos_emb_scale1 = row_emb_scale1 + col_emb_scale1  # (batch_size, context_length * tokens_per_grid, embed_dim//2)
        pos_emb_scale2 = row_emb_scale2 + col_emb_scale2  # (batch_size, context_length * tokens_per_grid, embed_dim//2)
        
        # Concatenate to form multi-scale positional encoding
        pos_embeddings = torch.cat([pos_emb_scale1, pos_emb_scale2], dim=-1)  # (batch_size, context_length * tokens_per_grid, embed_dim)
        
        # Get token embeddings
        token_embeddings = self.embedding(x)  # (batch_size, context_length * tokens_per_grid, embed_dim)
        
        # Add positional embeddings
        embeddings = token_embeddings + pos_embeddings  # (batch_size, context_length * tokens_per_grid, embed_dim)
        
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
        outputs = outputs.reshape(batch_size, context_length, self.max_tokens_per_grid, -1)
        
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

def train_model(model, train_loader, val_loader, num_epochs, device, is_vis):
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    
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
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    # Plot loss curves
    plot_train_val_curves(train_losses, val_losses)

def main(remap_colors=False, replace_colors=False, is_vis=False):
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
        padding_value=PADDING_VALUE,
        max_grid_size=MAX_GRID_SIZE
    ).to(device)

    if os.path.exists("tokenized_grid_transformer_finetuned.pth"):
        print("Loading pre-trained model...")
        state_dict = torch.load("tokenized_grid_transformer_finetuned.pth")
        # Load state_dict with strict=False to ignore missing and unexpected keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
    else:
        print("Pre-trained model not found. Training from scratch...")

    train_loader, val_loader = prepare_arc_data(
        "data/arc-agi_training_challenges.json",
        "data/arc-agi_evaluation_challenges.json",
        batch_size=BATCH_SIZE,
        padding_value=PADDING_VALUE,
        remap_colors=remap_colors,
        replace_colors=replace_colors
    )

    train_model(model, train_loader, val_loader, NUM_EPOCHS, device, is_vis)

    torch.save(model.state_dict(), "tokenized_grid_transformer_finetuned.pth")
    print("Final model saved as tokenized_grid_transformer_finetuned.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TokenizedGridTransformer model")
    parser.add_argument("--remap_colors", action="store_true", help="Remap colors to a canonical list")
    parser.add_argument("--replace_colors", action="store_true", help="Replace colors with random new colors")
    parser.add_argument("--is_vis", action="store_true", help="Enable visualization after training")
    args = parser.parse_args()

    main(remap_colors=args.remap_colors, replace_colors=args.replace_colors, is_vis=args.is_vis)

