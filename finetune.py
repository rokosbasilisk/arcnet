#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from tqdm import tqdm
import os

# Constants
GRID_SIZE = 30
NUM_COLORS = 10  # 0-9
CONTEXT_LENGTH = 4
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

# Import the GridTransformer model
from grid_transformer import GridTransformer

class ARCDataset(Dataset):
    def __init__(self, challenges_file, solutions_file):
        with open(challenges_file, 'r') as f:
            self.challenges = json.load(f)
        with open(solutions_file, 'r') as f:
            self.solutions = json.load(f)
        
        self.data = []
        for challenge_id in self.challenges:
            challenge = self.challenges[challenge_id]
            solution = self.solutions[challenge_id]
            
            for train_pair in challenge['train']:
                input_sequence = self.preprocess_grid(train_pair['input'])
                output_grid = self.preprocess_single_grid(train_pair['output'])
                
                # Ensure we have at least CONTEXT_LENGTH frames
                while len(input_sequence) < CONTEXT_LENGTH:
                    input_sequence.insert(0, np.zeros((GRID_SIZE, GRID_SIZE), dtype=int))
                
                # Use the last CONTEXT_LENGTH frames to predict the output
                input_window = input_sequence[-CONTEXT_LENGTH:]
                self.data.append((input_window, output_grid))
    
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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_grids, target_grid = self.data[idx]
        return torch.tensor(input_grids, dtype=torch.long), torch.tensor(target_grid, dtype=torch.long)

def exact_match_accuracy(outputs, targets):
    predicted = outputs.argmax(dim=1)
    correct = (predicted == targets).all(dim=(1, 2)).float()
    return correct.mean().item()

def cell_accuracy(outputs, targets):
    predicted = outputs.argmax(dim=1)
    correct = (predicted == targets).float()
    return correct.mean().item()

def fine_tune_model(model, train_loader, val_loader, num_epochs, device):
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
            loss = criterion(outputs.reshape(-1, NUM_COLORS), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_exact_acc += exact_match_accuracy(outputs, targets)
            train_cell_acc += cell_accuracy(outputs, targets)

        # Validation
        model.eval()
        val_loss = 0.0
        val_exact_acc = 0.0
        val_cell_acc = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.reshape(-1, NUM_COLORS), targets.reshape(-1))
                val_loss += loss.item()
                val_exact_acc += exact_match_accuracy(outputs, targets)
                val_cell_acc += cell_accuracy(outputs, targets)
        
        train_loss /= len(train_loader)
        train_exact_acc /= len(train_loader)
        train_cell_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_exact_acc /= len(val_loader)
        val_cell_acc /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Exact Acc: {train_exact_acc:.4f}, Train Cell Acc: {train_cell_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Exact Acc: {val_exact_acc:.4f}, Val Cell Acc: {val_cell_acc:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-trained model
    model = GridTransformer(num_layers=2, embed_dim=64, num_heads=4, ff_dim=256).to(device)
    model.load_state_dict(torch.load("grid_transformer_model_exact_loss.pth"))
    model = model.to(device)

    # Create dataset and dataloader for ARC data
    arc_dataset = ARCDataset("data/arc-agi_training_challenges.json", "data/arc-agi_training_solutions.json")
    train_size = int(0.8 * len(arc_dataset))
    val_size = len(arc_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(arc_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Fine-tune the model
    fine_tune_model(model, train_loader, val_loader, NUM_EPOCHS, device)

    # Save the fine-tuned model
    torch.save(model.state_dict(), "grid_transformer_finetuned.pth")
    print("Fine-tuned model saved as grid_transformer_finetuned.pth")

if __name__ == "__main__":
    main()
