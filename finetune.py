import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import json
import numpy as np
from tqdm import tqdm
import os
import random
from termcolor import colored
from tokenized_grid_transformer import TokenizedGridTransformer, ARCTokenizer, GRID_SIZE, CONTEXT_LENGTH

# Constants
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
NUM_LAYERS = 8
EMBED_DIM = 64
NUM_HEADS = 8
FF_DIM = 64
PADDING_VALUE = 10

import numpy as np
import torch
from torch.utils.data import Dataset
import json
import random

class ARCDataset(Dataset):
    def __init__(self, challenges_file, solutions_file, max_grid_size=30, context_length=CONTEXT_LENGTH, padding_value=10):
        with open(challenges_file, 'r') as f:
            self.challenges = json.load(f)
        with open(solutions_file, 'r') as f:
            self.solutions = json.load(f)
        
        self.tokenizer = ARCTokenizer(padding_value=padding_value)
        self.data = []
        self.max_grid_size = max_grid_size
        self.context_length = context_length
        self.padding_value = padding_value
        
        for challenge_id in self.challenges:
            challenge = self.challenges[challenge_id]
            solution = self.solutions[challenge_id]
            
            for train_pair in challenge['train']:
                input_sequence = self.preprocess_grid(train_pair['input'])
                output_grid = self.preprocess_single_grid(train_pair['output'])
                
                # Pad and tokenize input sequence and output grid
                input_tokens = [self.tokenize_and_pad(grid) for grid in input_sequence]
                output_tokens = self.tokenize_and_pad(output_grid)
                
                # Ensure we have exactly context_length frames
                if len(input_tokens) > self.context_length:
                    input_tokens = input_tokens[-self.context_length:]
                while len(input_tokens) < self.context_length:
                    input_tokens.insert(0, [self.tokenizer.padding_token] * (self.max_grid_size // 2) ** 2)
                
                self.data.append((input_tokens, output_tokens))
    
    def preprocess_grid(self, grid):
        if isinstance(grid[0], list):  # If it's already a 2D grid
            return [self.preprocess_single_grid(grid)]
        else:  # If it's a sequence of grids
            return [self.preprocess_single_grid(g) for g in grid]
    
    def preprocess_single_grid(self, grid):
        return np.array(grid, dtype=int)
    
    def tokenize_and_pad(self, grid):
        padded_grid = np.full((self.max_grid_size, self.max_grid_size), self.padding_value, dtype=int)
        h, w = grid.shape
        padded_grid[:h, :w] = grid
        return self.tokenizer.tokenize(padded_grid)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_tokens, target_tokens = self.data[idx]
        return (torch.tensor(input_tokens, dtype=torch.long),
                torch.tensor(target_tokens, dtype=torch.long))

def prepare_arc_data(train_file, eval_file, batch_size, padding_value=10, val_fraction=0.1):
    # Load both datasets
    full_dataset = ARCDataset(train_file, train_file.replace('challenges', 'solutions'), padding_value=padding_value)
    full_dataset.data.extend(ARCDataset(eval_file, eval_file.replace('challenges', 'solutions'), padding_value=padding_value).data)

    
    # Shuffle the combined dataset
    random.shuffle(full_dataset.data)
    
    # Calculate the split
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_fraction)
    train_size = dataset_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Total samples: {dataset_size}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    return train_loader, val_loader


def ensure_2d_grid(grid):
    if grid.ndim == 1:
        side_length = int(np.sqrt(grid.shape[0]))
        return grid.reshape(side_length, side_length)
    elif grid.ndim > 2:
        return grid.reshape(grid.shape[0], -1)
    return grid


def exact_match_accuracy(outputs, targets):
    predicted = outputs.argmax(dim=-1)
    correct = (predicted == targets).all(dim=-1).float()
    return correct.mean().item()

def cell_accuracy(outputs, targets):
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

        visualize_examples(model, val_loader, device)

        #torch.save(model.state_dict(), f"tokenized_grid_transformer_epoch_{epoch+1}.pth")



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

def visualize_examples(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(val_loader))
        idx = random.randint(0, targets.size(0) - 1)
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
        
        predicted_grid = model.tokenizer.detokenize(predicted_sample.cpu().numpy(), (GRID_SIZE, GRID_SIZE))
        target_grid = model.tokenizer.detokenize(target_sample.cpu().numpy(), (GRID_SIZE, GRID_SIZE))
        
        print(colored("\nGround Truth vs Predicted", 'yellow'))
        print(f"Target shape: {target_grid.shape}, Predicted shape: {predicted_grid.shape}")
        print_grid(target_grid, predicted_grid)
        
        print("\n" + "-" * 40 + "\n")  # Separator between examples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TokenizedGridTransformer(
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        max_seq_length=CONTEXT_LENGTH * (GRID_SIZE // 2)**2,
        padding_value=PADDING_VALUE
    ).to(device)

    if os.path.exists("tokenized_grid_transformer_finetuned.pth"):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load("tokenized_grid_transformer_finetuned.pth"))
    else:
        print("Pre-trained model not found. Training from scratch...")

    train_loader, val_loader = prepare_arc_data(
        "data/arc-agi_training_challenges.json",
        "data/arc-agi_evaluation_challenges.json",
        BATCH_SIZE,
        padding_value=PADDING_VALUE
    )

    train_model(model, train_loader, val_loader, NUM_EPOCHS, device)

    torch.save(model.state_dict(), "tokenized_grid_transformer_finetuned.pth")
    print("Final model saved as tokenized_grid_transformer_finetuned.pth")

if __name__ == "__main__":
    main()
