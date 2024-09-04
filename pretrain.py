import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from grid_transformer import *
import json
import os
import pygame
import numpy as np
from tqdm import tqdm
import time
from vis import *

# Constants
GRID_SIZE = 30
NUM_COLORS = 10  # 0-9
CONTEXT_LENGTH = 6
BATCH_SIZE = 6
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_LAYERS = 3

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)


# Constants
GRID_SIZE = 30
CELL_SIZE = 20
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
FPS = 30

# Color mapping
COLOR_MAP = [
    (0, 0, 0),      # 0: Black (empty)
    (255, 0, 0),    # 1: Red
    (0, 255, 0),    # 2: Green
    (0, 0, 255),    # 3: Blue
    (255, 255, 0),  # 4: Yellow
    (255, 0, 255),  # 5: Magenta
    (0, 255, 255),  # 6: Cyan
    (128, 0, 0),    # 7: Maroon
    (0, 128, 0),    # 8: Dark Green
    (0, 0, 128)     # 9: Navy
]



class GridDataset(Dataset):
    def __init__(self, trajectories):
        self.data = []
        for trajectory in trajectories:
            states = [state["state"] for state in trajectory["trajectory"]]
            for i in range(len(states) - CONTEXT_LENGTH - 1):
                input_states = states[i:i+CONTEXT_LENGTH]
                target_state = states[i+CONTEXT_LENGTH]
                self.data.append((input_states, target_state))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_states, target_state = self.data[idx]
        input_tensor = torch.tensor(input_states, dtype=torch.long)
        target_tensor = torch.tensor(target_state, dtype=torch.long)
        return input_tensor, target_tensor


def load_trajectories(folder_path, num_trajectories=200):
    trajectories = []
    for i in range(1, num_trajectories + 1):
        file_path = os.path.join(folder_path, f"trajectory-{i}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                trajectories.append(json.load(f))
    return trajectories

def exact_match_accuracy(outputs, targets):
    predicted = outputs.argmax(dim=1)
    correct = (predicted == targets).all(dim=(1, 2)).float()
    return correct.mean().item()

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    trajectories = load_trajectories("trajectories")
    train_size = int(0.8 * len(trajectories))
    train_trajectories = trajectories[:train_size]
    val_trajectories = trajectories[train_size:]

    train_dataset = GridDataset(train_trajectories)
    val_dataset = GridDataset(val_trajectories)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = GridTransformer(num_layers=NUM_LAYERS, embed_dim=64, num_heads=4, ff_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, NUM_COLORS), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += exact_match_accuracy(outputs, targets)

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.reshape(-1, NUM_COLORS), targets.reshape(-1))
                val_loss += loss.item()
                val_acc += exact_match_accuracy(outputs, targets)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Visualize examples after each epoch
        visualize_examples(model, val_loader, device)

    # Save the model
    torch.save(model.state_dict(), "grid_transformer_model_exact_loss.pth")
    print("Model saved as grid_transformer_model_exact_loss.pth")

if __name__ == "__main__":
    train_model()

