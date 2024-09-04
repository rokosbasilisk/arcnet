import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from tqdm import tqdm
import os
import pygame
import random

# Constants
GRID_SIZE = 30
NUM_COLORS = 10  # 0-9
CONTEXT_LENGTH = 6
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_LAYERS = 3
CELL_SIZE = 20
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

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


# Constants
GRID_SIZE = 30
CELL_SIZE = 20
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
FPS = 30

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

def draw_grid(screen, grid_state, x_offset=0):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = COLOR_MAP[grid_state[i][j]]
            pygame.draw.rect(screen, color, (x_offset + j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, GRAY, (x_offset + j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

def visualize_examples(model, val_loader, device):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE * 2, SCREEN_SIZE))
    pygame.display.set_caption("Ground Truth vs Predicted")
    clock = pygame.time.Clock()

    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)

            for i in range(5):  # Display 5 random examples
                idx = random.randint(0, targets.size(0) - 1)
                
                screen.fill(BLACK)
                draw_grid(screen, targets[idx].cpu().numpy())
                draw_grid(screen, predicted[idx].cpu().numpy(), x_offset=SCREEN_SIZE)

                font = pygame.font.Font(None, 36)
                gt_text = font.render("Ground Truth", True, WHITE)
                pred_text = font.render("Predicted", True, WHITE)
                screen.blit(gt_text, (10, 10))
                screen.blit(pred_text, (SCREEN_SIZE + 10, 10))

                pygame.display.flip()
                clock.tick(FPS)

                # Wait for a key press or quit event
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                        if event.type == pygame.KEYDOWN:
                            waiting = False

    pygame.quit()

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

        # Visualize examples after each epoch
        visualize_examples(model, val_loader, device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = GridTransformer(num_layers=NUM_LAYERS, embed_dim=64, num_heads=4, ff_dim=256).to(device)

    # Check if pre-trained model exists
    if os.path.exists("grid_transformer_model_exact_loss.pth"):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load("grid_transformer_model_exact_loss.pth"))
    else:
        print("Pre-trained model not found. Training from scratch...")

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
