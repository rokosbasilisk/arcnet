#from vis import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from tqdm import tqdm
import os
import random
from termcolor import colored
from tokenized_grid_transformer import TokenizedGridTransformer, ARCTokenizer, GRID_SIZE, CONTEXT_LENGTH, NUM_COLORS

# Constants
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_LAYERS = 5
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 1024



import numpy as np
import torch
from torch.utils.data import Dataset
import json
import random

class ARCDataset(Dataset):
    def __init__(self, challenges_file, solutions_file, apply_transformations=True):
        with open(challenges_file, 'r') as f:
            self.challenges = json.load(f)
        with open(solutions_file, 'r') as f:
            self.solutions = json.load(f)
        
        self.tokenizer = ARCTokenizer()
        self.data = []
        self.apply_transformations = apply_transformations
        
        for challenge_id in self.challenges:
            challenge = self.challenges[challenge_id]
            solution = self.solutions[challenge_id]
            
            for train_pair in challenge['train']:
                input_sequence = self.preprocess_grid(train_pair['input'])
                output_grid = self.preprocess_single_grid(train_pair['output'])
                
                if self.apply_transformations:
                    input_sequence, output_grid = self.apply_random_transformations(input_sequence, output_grid)
                
                # Tokenize input sequence and output grid
                input_tokens = [self.tokenizer.tokenize(self.tokenizer.pad_grid(grid)) for grid in input_sequence]
                output_tokens = self.tokenizer.tokenize(self.tokenizer.pad_grid(output_grid))
                
                # Ensure we have exactly CONTEXT_LENGTH frames
                if len(input_tokens) > CONTEXT_LENGTH:
                    input_tokens = input_tokens[-CONTEXT_LENGTH:]
                while len(input_tokens) < CONTEXT_LENGTH:
                    input_tokens.insert(0, [0] * ((GRID_SIZE // 2) ** 2))
                
                self.data.append((input_tokens, output_tokens))
    
    def preprocess_grid(self, grid):
        if isinstance(grid[0], list):  # If it's already a 2D grid
            return [self.preprocess_single_grid(grid)]
        else:  # If it's a sequence of grids
            return [self.preprocess_single_grid(g) for g in grid]
    
    def preprocess_single_grid(self, grid):
        padded_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        h, w = min(GRID_SIZE, len(grid)), min(GRID_SIZE, len(grid[0]))
        padded_grid[:h, :w] = np.array(grid)[:h, :w]
        return padded_grid
    
    def apply_random_transformations(self, input_sequence, output_grid):
        # Randomly choose transformations
        transformations = []
        if random.random() < 0.5:
            transformations.append(self.rotate_grid)
        if random.random() < 0.5:
            transformations.append(self.replace_colors)
        
        # Apply chosen transformations
        for transform in transformations:
            input_sequence, output_grid = transform(input_sequence, output_grid)
        
        return input_sequence, output_grid
    
    def rotate_grid(self, input_sequence, output_grid):
        k = random.choice([1, 2, 3])  # Number of 90-degree rotations
        rotated_input = [np.rot90(grid, k) for grid in input_sequence]
        rotated_output = np.rot90(output_grid, k)
        return rotated_input, rotated_output
    
    def replace_colors(self, input_sequence, output_grid):
        # Get unique colors excluding 0
        unique_colors = set(np.unique(input_sequence + [output_grid])) - {0}
        
        # Create color map, keeping 0 unchanged
        color_map = {0: 0}  # Ensure 0 maps to 0
        available_colors = list(range(1, NUM_COLORS))
        for color in unique_colors:
            if color != 0:
                new_color = random.choice(available_colors)
                color_map[color] = new_color
                available_colors.remove(new_color)
        
        # Apply color replacement
        replaced_input = [np.vectorize(lambda x: color_map.get(x, x))(grid) for grid in input_sequence]
        replaced_output = np.vectorize(lambda x: color_map.get(x, x))(output_grid)
        
        return replaced_input, replaced_output
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_tokens, target_tokens = self.data[idx]
        return (torch.tensor(input_tokens, dtype=torch.long),
                torch.tensor(target_tokens, dtype=torch.long))


def color_for_value(value):
    """ Returns a color based on the value for grid cells ranging from 0 to 9. """
    color_map = {
        0: 'white',
        1: 'blue',
        2: 'cyan',
        3: 'magenta',
        4: 'yellow',
        5: 'green',
        6: 'red',
        7: 'grey',
        8: 'light_blue',
        9: 'light_magenta'
    }
    return color_map.get(value, 'white')  # Default to 'white' if value is not found

def ensure_2d_grid(grid):
    """Ensure the grid is a 2D numpy array."""
    if grid.ndim == 1:
        side_length = int(np.sqrt(grid.shape[0]))
        return grid.reshape(side_length, side_length)
    elif grid.ndim > 2:
        return grid.reshape(grid.shape[0], -1)
    return grid

def print_grid(gt_grid, pred_grid):
    """ Helper function to print two grids side by side in the terminal. """
    for gt_row, pred_row in zip(gt_grid, pred_grid):
        gt_line = []
        pred_line = []
        for gt_cell, pred_cell in zip(gt_row, pred_row):
            # Get colored strings based on cell values
            gt_line.append(colored(str(int(gt_cell)), color_for_value(int(gt_cell))))
            pred_line.append(colored(str(int(pred_cell)), color_for_value(int(pred_cell))))
        # Print the two rows side by side
        print(" ".join(gt_line) + "    " + " ".join(pred_line))


def exact_match_accuracy(outputs, targets):
    predicted = outputs.argmax(dim=-1)
    correct = (predicted == targets).all(dim=-1).float()
    return correct.mean().item()

def cell_accuracy(outputs, targets):
    predicted = outputs.argmax(dim=-1)
    correct = (predicted == targets).float()
    return correct.mean().item()

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_exact_acc = 0.0
        train_cell_acc = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs.reshape(-1, model.tokenizer.vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_exact_acc += exact_match_accuracy(outputs, targets)
            train_cell_acc += cell_accuracy(outputs, targets)

        train_loss /= len(train_loader)
        train_exact_acc /= len(train_loader)
        train_cell_acc /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_exact_acc = 0.0
        val_cell_acc = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.reshape(-1, model.tokenizer.vocab_size), targets.reshape(-1))
                val_loss += loss.item()
                val_exact_acc += exact_match_accuracy(outputs, targets)
                val_cell_acc += cell_accuracy(outputs, targets)
        
        val_loss /= len(val_loader)
        val_exact_acc /= len(val_loader)
        val_cell_acc /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Exact Acc: {train_exact_acc:.4f}, Train Cell Acc: {train_cell_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Exact Acc: {val_exact_acc:.4f}, Val Cell Acc: {val_cell_acc:.4f}")

        # Visualize examples after each epoch
        visualize_examples(model, val_loader, device)

        # Save the model after each epoch
        #torch.save(model.state_dict(), f"tokenized_grid_transformer_epoch_{epoch+1}.pth")

def visualize_examples(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(val_loader))
        idx = random.randint(0, targets.size(0) - 1)
        input_sample = inputs[idx].unsqueeze(0).to(device)
        target_sample = targets[idx].to(device)
        
        output_sample = model(input_sample)
        
        #print("Raw model output shape:", output_sample.shape)
        #print("Raw model output min:", output_sample.min().item())
        #print("Raw model output max:", output_sample.max().item())
        
        # Use temperature sampling
        temperature = 0.8
        output_sample = output_sample / temperature
        output_probs = torch.softmax(output_sample, dim=-1)
        predicted_sample = torch.multinomial(output_probs.view(-1, output_probs.size(-1)), num_samples=1).squeeze()
        
        #print("Predicted sample shape:", predicted_sample.shape)
        #print("Predicted sample min:", predicted_sample.min().item())
        #print("Predicted sample max:", predicted_sample.max().item())
        
        # Debug: Print unique predicted tokens and their counts
        unique_tokens, counts = torch.unique(predicted_sample, return_counts=True)
        print("Unique predicted tokens:", unique_tokens.tolist())
        print("Token counts:", counts.tolist())
        
        # Detokenize the predicted sample
        predicted_grid = model.tokenizer.detokenize(predicted_sample.cpu().numpy(), (GRID_SIZE, GRID_SIZE))
        predicted_grid = ensure_2d_grid(predicted_grid)
        
        # Detokenize the target sample
        target_grid = model.tokenizer.detokenize(target_sample.cpu().numpy(), (GRID_SIZE, GRID_SIZE))
        target_grid = ensure_2d_grid(target_grid)
        
        print(colored("Ground Truth vs Predicted", 'yellow'))
        print(f"Target shape: {target_grid.shape}, Predicted shape: {predicted_grid.shape}")
        print_grid(target_grid, predicted_grid)
        
        print("\n" + "-" * 20 + "\n")  # Separator between examples



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = TokenizedGridTransformer(
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        max_seq_length=CONTEXT_LENGTH * (GRID_SIZE // 2)**2
    ).to(device)

    # Check if pre-trained model exists
    if os.path.exists("tokenized_grid_transformer_finetuned.pth"):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load("tokenized_grid_transformer_finetuned.pth"))
    else:
        print("Pre-trained model not found. Training from scratch...")

    # Create dataset and dataloader for ARC data
    train_dataset = ARCDataset("data/arc-agi_training_challenges.json", "data/arc-agi_training_solutions.json")
    val_dataset = ARCDataset("data/arc-agi_evaluation_challenges.json", "data/arc-agi_evaluation_solutions.json")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Train the model
    train_model(model, train_loader, val_loader, NUM_EPOCHS, device)

    # Save the final model
    torch.save(model.state_dict(), "tokenized_grid_transformer_finetuned.pth")
    print("Final model saved as tokenized_grid_transformer_finetuned.pth")

if __name__ == "__main__":
    main()
