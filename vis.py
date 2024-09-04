import pygame
import json
import sys
import os
from termcolor import colored
import random
import torch

# Constants
GRID_SIZE = 30
CELL_SIZE = 20
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

# Color mapping (for visualization purposes only)
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

def load_trajectory(folder_path, trajectory_number):
    file_path = os.path.join(folder_path, f"trajectory-{trajectory_number}.json")
    with open(file_path, 'r') as f:
        return json.load(f)

def draw_grid(screen, grid_state):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = COLOR_MAP[grid_state[i][j]]
            pygame.draw.rect(screen, color, (j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, GRAY, (j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

def main(trajectory_folder, trajectory_number):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption(f"Trajectory Visualizer - Trajectory {trajectory_number}")
    clock = pygame.time.Clock()

    trajectory_data = load_trajectory(trajectory_folder, trajectory_number)
    trajectory = trajectory_data["trajectory"]
    current_state_index = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    current_state_index = min(current_state_index + 1, len(trajectory) - 1)
                elif event.key == pygame.K_LEFT:
                    current_state_index = max(current_state_index - 1, 0)

        screen.fill(BLACK)
        draw_grid(screen, trajectory[current_state_index]["state"])

        font = pygame.font.Font(None, 36)
        state_text = font.render(f"State: {current_state_index + 1}/{len(trajectory)}", True, WHITE)
        move_text = font.render(f"Move: {trajectory[current_state_index]['move']}", True, WHITE)
        screen.blit(state_text, (10, 10))
        screen.blit(move_text, (10, 50))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

def visualize_examples(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        for _ in range(3):  # Display only 3 examples
            # Select a random batch from the loader
            inputs, targets = next(iter(val_loader))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)

            # Pick a random example from the batch
            idx = random.randint(0, targets.size(0) - 1)
            
            print(colored("Ground Truth vs Predicted", 'yellow'))
            print_grid(targets[idx].cpu().numpy(), predicted[idx].cpu().numpy())
            
            print("\n" + "-" * 20 + "\n")  # Separator between examples

def print_grid(gt_grid, pred_grid):
    """ Helper function to print two grids side by side in the terminal. """
    for gt_row, pred_row in zip(gt_grid, pred_grid):
        gt_line = []
        pred_line = []

        for gt_cell, pred_cell in zip(gt_row, pred_row):
            # Get colored strings based on cell values
            gt_line.append(colored(str(gt_cell), color_for_value(gt_cell)))
            pred_line.append(colored(str(pred_cell), color_for_value(pred_cell)))

        # Print the two rows side by side
        print(" ".join(gt_line) + "    " + " ".join(pred_line))

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
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python visualizer.py <trajectory_folder> <trajectory_number>")
        sys.exit(1)

    trajectory_folder = sys.argv[1]
    trajectory_number = int(sys.argv[2])
    main(trajectory_folder, trajectory_number)
