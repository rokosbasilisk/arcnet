import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
import os
import random

# Constants
GRID_SIZE = 30  # Maximum grid size (30x30)
RELATIONS = ['NOR', 'NCONV', 'NIMPL', 'AND']
PADDING_VALUE = 10
CONTEXT_LENGTH = 6  # Number of previous grids to consider
NUM_EPOCHS = 10
BATCH_SIZE = 1  # Reduce batch size to prevent OOM
EMBED_DIM = 32  # Reduced embedding dimension
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128
VALIDATION_SPLIT = 0.2  # 20% for validation

# Ensure output directory exists for plots
OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HNet-Inspired Hamiltonian Layer
class HamiltonianLayer(nn.Module):
    def __init__(self, embed_dim, relations=RELATIONS):
        super(HamiltonianLayer, self).__init__()
        self.embed_dim = embed_dim
        self.relations = relations
        self.num_relations = len(relations)
        
        # Initialize Hamiltonians for each relation as learnable parameters
        # Shape: (num_relations, embed_dim, embed_dim)
        self.hamiltonians = nn.Parameter(torch.randn(self.num_relations, embed_dim, embed_dim))
    
    def forward(self, token_embeddings, relation_indices):
        """
        token_embeddings: (batch_size, seq_length, embed_dim)
        relation_indices: (batch_size, seq_length, seq_length) with values in [0, num_relations-1]
        Returns:
            energy: (batch_size, seq_length, seq_length)
        """
        batch_size, seq_length, embed_dim = token_embeddings.size()
        device = token_embeddings.device
        energy = torch.zeros(batch_size, seq_length, seq_length, device=device)
        
        # Iterate over each relation type
        for rel_idx, relation in enumerate(self.relations):
            # Get mask for current relation
            mask = (relation_indices == rel_idx).float()  # (batch_size, seq_length, seq_length)
            if mask.sum() == 0:
                continue  # Skip if no such relations in the batch
            
            # Get corresponding Hamiltonian
            H = self.hamiltonians[rel_idx]  # (embed_dim, embed_dim)
            
            # Compute energy: e_ij = e_i^T * H * e_j
            # token_embeddings: (batch_size, seq_length, embed_dim)
            # Using einsum for efficient batch computation
            # Equation: 'bik,kl,bjk->bij'
            energy_rel = torch.einsum('bik,kl,bjk->bij', token_embeddings, H, token_embeddings)  # (batch_size, seq_length, seq_length)
            energy += energy_rel * mask
        
        return energy  # (batch_size, seq_length, seq_length)

# ARCTokenizer Class
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

# ARCDataset Class
class ARCDataset(Dataset):
    def __init__(self, challenges_file, solutions_file, max_grid_size=30, context_length=6, 
                 padding_value=10, remap_colors=False, replace_colors=False):
        """
        challenges_file: Path to the JSON file containing training/evaluation challenges.
        solutions_file: Path to the JSON file containing corresponding solutions.
        max_grid_size: Maximum grid size to pad or crop grids.
        context_length: Number of previous grids to consider (if applicable).
        padding_value: Value used for padding grids.
        remap_colors: Whether to remap colors to a limited range.
        replace_colors: Whether to replace colors with random values from a limited range.
        """
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
        self.max_tokens_per_grid = (self.max_grid_size // 2) ** 2
        
        self.process_data()

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

    def construct_relation_matrix_static(self, grid):
        """
        Construct a relation matrix for a given grid.
        """
        subgrid_size = 2
        tokens_per_row = grid.shape[1] // subgrid_size
        tokens_per_col = grid.shape[0] // subgrid_size
        seq_length = tokens_per_row * tokens_per_col
        
        relation_matrix = torch.full((seq_length, seq_length), self.tokenizer.padding_token, dtype=torch.long)
        
        # Define relations based on adjacent tokens (right and down)
        for i in range(tokens_per_col):
            for j in range(tokens_per_row):
                idx = i * tokens_per_row + j
                # Right relation
                if j < tokens_per_row - 1:
                    right_idx = i * tokens_per_row + (j + 1)
                    # Determine relation based on subgrids
                    relation = self.determine_relation(grid, i, j, i, j+1)
                    relation_matrix[idx, right_idx] = relation
                # Down relation
                if i < tokens_per_col - 1:
                    down_idx = (i + 1) * tokens_per_row + j
                    # Determine relation based on subgrids
                    relation = self.determine_relation(grid, i, j, i+1, j)
                    relation_matrix[idx, down_idx] = relation
        return relation_matrix

    def determine_relation(self, grid, i1, j1, i2, j2):
        """
        Determine the relation type between two adjacent subgrids.
        """
        subgrid_size = 2
        # Extract subgrids
        subgrid1 = grid[i1*subgrid_size:(i1+1)*subgrid_size, j1*subgrid_size:(j1+1)*subgrid_size].flatten()
        subgrid2 = grid[i2*subgrid_size:(i2+1)*subgrid_size, j2*subgrid_size:(j2+1)*subgrid_size].flatten()
        # Simple relation based on sum of subgrid values
        sum1 = subgrid1.sum()
        sum2 = subgrid2.sum()
        if sum1 == 0 and sum2 == 0:
            return 0  # 'NOR'
        elif sum1 == 0 and sum2 > 0:
            return 1  # 'NCONV'
        elif sum1 > 0 and sum2 == 0:
            return 2  # 'NIMPL'
        elif sum1 > 0 and sum2 > 0:
            return 3  # 'AND'
        else:
            return 0  # Default to 'NOR'

    def process_data(self):
        for challenge_id in self.challenges:
            challenge = self.challenges[challenge_id]
            
            for idx, train_pair in enumerate(challenge.get('train', [])):
                input_sequence = self.preprocess_grid(train_pair['input'])  # List of grids
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
                # Pad or truncate output tokens to match input length
                output_tokens = output_tokens[:self.max_tokens_per_grid] + [self.tokenizer.padding_token] * (self.max_tokens_per_grid - len(output_tokens))
                
                # Construct relation_matrix for the entire sequence
                relation_matrices = [self.construct_relation_matrix_static(grid) for grid in augmented_input_sequence]
                relation_matrix = torch.block_diag(*relation_matrices)  # (context_length * tokens_per_grid, context_length * tokens_per_grid)
                
                self.data.append((input_tokens, output_tokens, relation_matrix))
        
        print(f"Total loaded training examples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tokens, output_tokens, relation_matrix = self.data[idx]
        return (torch.tensor(input_tokens, dtype=torch.long),
                torch.tensor(output_tokens, dtype=torch.long),
                relation_matrix.clone().detach())

# GridTransformerWithHNet Class
class GridTransformerWithHNet(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, vocab_size, padding_value=10):
        super(GridTransformerWithHNet, self).__init__()
        self.tokenizer = ARCTokenizer(padding_value=padding_value)
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.tokenizer.padding_token)
        
        # Positional Embedding
        self.pos_embedding = nn.Embedding(10000, embed_dim)  # Adjust max sequence length as needed
        
        # Hamiltonian Layer
        self.hamiltonian_layer = HamiltonianLayer(embed_dim=embed_dim, relations=RELATIONS)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final Linear Layer
        self.final_layer = nn.Linear(embed_dim, vocab_size)
        
        self.padding_value = padding_value
        
    def forward(self, x, relation_indices):
        """
        x: (batch_size, context_length, tokens_per_grid)
        relation_indices: (batch_size, seq_length, seq_length)
        """
        batch_size, context_length, tokens_per_grid = x.shape
        x = x.view(batch_size, -1)  # Flatten to (batch_size, seq_length)
        seq_length = x.size(1)
        
        # Get token embeddings
        token_embeddings = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        
        # Generate positional embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_embeddings = self.pos_embedding(position_ids)  # (seq_length, embed_dim)
        embeddings = token_embeddings + position_embeddings.unsqueeze(0)  # (batch_size, seq_length, embed_dim)
        
        # Prepare for Transformer: (seq_length, batch_size, embed_dim)
        embeddings = embeddings.transpose(0, 1)
        
        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(embeddings)  # (seq_length, batch_size, embed_dim)
        
        # Back to (batch_size, seq_length, embed_dim)
        transformer_output = transformer_output.transpose(0, 1)
        
        # Reshape relation_indices to (batch_size, seq_length, seq_length)
        relation_indices = relation_indices.view(batch_size, seq_length, seq_length)
        
        # Apply Hamiltonian Layer
        energy = self.hamiltonian_layer(transformer_output, relation_indices)  # (batch_size, seq_length, seq_length)
        
        # Aggregate energy across related tokens
        energy_sum = energy.sum(dim=2)  # (batch_size, seq_length)
        
        # Final logits
        logits = self.final_layer(transformer_output) + energy_sum.unsqueeze(-1)  # (batch_size, seq_length, vocab_size)
        
        # Only return the prediction for the last frame
        outputs = logits[:, -tokens_per_grid:, :]  # (batch_size, tokens_per_grid, vocab_size)
        
        return outputs

    def generate(self, input_sequence, relation_indices, max_new_tokens=1):
        self.eval()
        generated = input_sequence
        relation = relation_indices
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(generated, relation)  # (batch_size, tokens_per_grid, vocab_size)
                next_token = outputs.argmax(dim=-1).unsqueeze(1)  # (batch_size, 1, tokens_per_grid)
                # Shift input and append the new token
                generated = torch.cat([generated[:, 1:, :], next_token], dim=1)
        return generated[:, -1, :]  # (batch_size, tokens_per_grid, vocab_size)

# Visualization Function
def visualize_grid(grid, title="Grid", save_path=None):
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Training Function with Validation
def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        epoch_loss = 0.0
        for inputs, targets, relation_matrix in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, targets, relation_matrix = inputs.to(device), targets.to(device), relation_matrix.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, relation_matrix)  # (batch_size, tokens_per_grid, vocab_size)
            
            # Reshape for loss computation
            outputs = outputs.view(-1, model.tokenizer.vocab_size)
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, relation_matrix in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, targets, relation_matrix = inputs.to(device), targets.to(device), relation_matrix.to(device)
                
                outputs = model(inputs, relation_matrix)  # (batch_size, tokens_per_grid, vocab_size)
                
                # Reshape for loss computation
                outputs = outputs.view(-1, model.tokenizer.vocab_size)
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Scheduler Step
        scheduler.step(avg_val_loss)
        
        # Print Losses
        print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
    
    # Plot loss curves and save to file
    plt.figure()
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'))
    plt.close()
    print(f"Loss curves saved to {os.path.join(OUTPUT_DIR, 'loss_curves.png')}")

# Prediction and Visualization Function
def predict_and_visualize(model, tokenizer, test_input_grid, test_output_grid, device):
    model.eval()
    with torch.no_grad():
        # Tokenize input grids
        tokens_input_sequence = []
        for i in range(CONTEXT_LENGTH):
            if i < len(test_input_grid):
                tokens = tokenizer.tokenize(test_input_grid[i])
            else:
                tokens = [tokenizer.padding_token] * tokenizer.max_tokens_per_grid
            tokens = tokens[:tokenizer.max_tokens_per_grid] + [tokenizer.padding_token] * (tokenizer.max_tokens_per_grid - len(tokens))
            tokens_input_sequence.append(tokens)
        tokens_input = np.array(tokens_input_sequence, dtype=np.int64)  # (context_length, tokens_per_grid)
        tokens_input_tensor = torch.tensor(tokens_input, dtype=torch.long).unsqueeze(0).to(device)  # (1, context_length, tokens_per_grid)
        
        # Construct relation matrix
        relation_matrices = [tokenizer.tokenize(tokenizer.pad_grid(grid)) for grid in test_input_grid]
        relation_matrices = [torch.tensor(tokenizer.tokenize(grid), dtype=torch.long) for grid in test_input_grid]
        relation_matrix = torch.block_diag(*relation_matrices).unsqueeze(0).to(device)  # (1, context_length * tokens_per_grid, context_length * tokens_per_grid)
        
        # Generate prediction
        outputs = model(tokens_input_tensor, relation_matrix)  # (1, tokens_per_grid, vocab_size)
        predicted_tokens = outputs.argmax(dim=-1).squeeze(0)  # (tokens_per_grid)
        
        # Decode tokens to grid
        predicted_grid = tokenizer.detokenize(predicted_tokens.cpu().numpy(), GRID_SIZE)
        true_grid = test_output_grid
        input_grid = test_input_grid[-1] if len(test_input_grid) > 0 else np.full((GRID_SIZE, GRID_SIZE), PADDING_VALUE)
        
        # Save visualizations
        visualize_grid(input_grid, title="Test Input Grid", save_path=os.path.join(OUTPUT_DIR, 'test_input_grid.png'))
        visualize_grid(true_grid, title="True Output Grid", save_path=os.path.join(OUTPUT_DIR, 'true_output_grid.png'))
        visualize_grid(predicted_grid, title="Predicted Output Grid", save_path=os.path.join(OUTPUT_DIR, 'predicted_output_grid.png'))
        print(f"Visualization images saved to {OUTPUT_DIR} directory.")

# Main Execution Script
def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Tokenizer
    tokenizer = ARCTokenizer(padding_value=PADDING_VALUE)
    
    # File paths
    training_challenges_file = 'data/arc-agi_training_challenges.json'
    training_solutions_file = 'data/arc-agi_training_solutions.json'
    evaluation_challenges_file = 'data/arc-agi_evaluation_challenges.json'
    evaluation_solutions_file = 'data/arc-agi_evaluation_solutions.json'
    
    # Check if files exist
    for file in [training_challenges_file, training_solutions_file, evaluation_challenges_file, evaluation_solutions_file]:
        if not os.path.exists(file):
            print(f"Error: File '{file}' does not exist. Please ensure the file path is correct.")
            return
    
    # Initialize Datasets
    train_dataset = ARCDataset(challenges_file=training_challenges_file,
                               solutions_file=training_solutions_file,
                               padding_value=PADDING_VALUE)
    
    val_dataset = ARCDataset(challenges_file=evaluation_challenges_file,
                             solutions_file=evaluation_solutions_file,
                             padding_value=PADDING_VALUE)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Check if datasets are not empty
    if len(train_dataset) == 0:
        print("Error: Training dataset is empty. Please check the training JSON files.")
        return
    if len(val_dataset) == 0:
        print("Error: Validation dataset is empty. Please check the evaluation JSON files.")
        return
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    model = GridTransformerWithHNet(
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        vocab_size=tokenizer.vocab_size,
        padding_value=PADDING_VALUE
    ).to(device)
    
    # Train the Model
    print("\nStarting Training...")
    train_model(model, train_loader, val_loader, NUM_EPOCHS, device)
    
    # Define Test Input (using a sample from validation set)
    # For demonstration, pick the first example from validation set
    if len(val_dataset) > 0:
        test_input_tokens, test_output_tokens, _ = val_dataset[0]
        test_input_sequence = []
        for tokens in test_input_tokens:
            grid = tokenizer.detokenize(tokens.numpy(), GRID_SIZE)
            test_input_sequence.append(grid)
        test_output_grid = tokenizer.detokenize(test_output_tokens.numpy(), GRID_SIZE)
        
        # Prediction and Visualization
        predict_and_visualize(model, tokenizer, test_input_sequence, test_output_grid, device)
    else:
        print("No validation samples available for prediction.")

if __name__ == "__main__":
    main()

