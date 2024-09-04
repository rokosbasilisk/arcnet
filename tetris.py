import pygame
import random
import sys
import json

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 30
CELL_SIZE = 20
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

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

# Updated Tetromino shapes with color information
SHAPES = [
    {"shape": [[1, 1, 1, 1]], "colors": [1]},  # I
    {"shape": [[1, 1], [1, 1]], "colors": [2]},  # O
    {"shape": [[1, 1, 1], [0, 1, 0]], "colors": [3]},  # T
    {"shape": [[1, 1, 1], [1, 0, 0]], "colors": [4]},  # L
    {"shape": [[1, 1, 1], [0, 0, 1]], "colors": [5]},  # J
    {"shape": [[1, 1, 0], [0, 1, 1]], "colors": [6]},  # S
    {"shape": [[0, 1, 1], [1, 1, 0]], "colors": [7]},  # Z
    # New complex shapes
    {"shape": [[1, 1, 1, 1], [1, 1, 1, 1]], "colors": [8]},  # Opaque rectangle
    {"shape": [[1, 1, 1], [1, 1, 1], [1, 1, 1]], "colors": [1, 2, 3]},  # Multi-colored 3x3 square
    {"shape": [[1, 1, 1, 1], [0, 1, 1, 0]], "colors": [4, 5]},  # Two-colored rectangle
    {"shape": [[1, 0, 1], [1, 1, 1], [1, 0, 1]], "colors": [6, 7]}  # Cross shape
]

def generate_random_shape():
    shape_type = random.choice([
        "I", "O", "T", "L", "J", "S", "Z",
        "rectangle", "square", "two_colored_rectangle", "cross"
    ])
    
    if shape_type == "I":
        length = random.randint(2, 6)
        shape = [[1] * length]
    elif shape_type == "O":
        size = random.randint(2, 4)
        shape = [[1] * size for _ in range(size)]
    elif shape_type == "T":
        width = random.randint(3, 5)
        shape = [[1] * width, [0] + [1] * (width-2) + [0]]
    elif shape_type in ["L", "J"]:
        height = random.randint(2, 4)
        shape = [[1] * 3 for _ in range(height)]
        if shape_type == "L":
            shape[-1] = [1, 0, 0]
        else:
            shape[-1] = [0, 0, 1]
    elif shape_type in ["S", "Z"]:
        width = random.randint(2, 4)
        if shape_type == "S":
            shape = [[0] * width + [1] * width, [1] * width + [0] * width]
        else:
            shape = [[1] * width + [0] * width, [0] * width + [1] * width]
    elif shape_type == "rectangle":
        width = random.randint(2, 5)
        height = random.randint(2, 3)
        shape = [[1] * width for _ in range(height)]
    elif shape_type == "square":
        size = random.randint(2, 4)
        shape = [[1] * size for _ in range(size)]
    elif shape_type == "two_colored_rectangle":
        width = random.randint(3, 5)
        shape = [[1] * width, [0] + [1] * (width-2) + [0]]
    elif shape_type == "cross":
        size = random.randint(3, 5)
        shape = [[0] * size for _ in range(size)]
        for i in range(size):
            shape[i][size//2] = 1
            shape[size//2][i] = 1
    
    num_colors = random.randint(1, 3)
    colors = random.sample(range(1, 10), num_colors)
    
    return {"shape": shape, "colors": colors}

class Tetris:
    def __init__(self):
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.current_piece = None
        self.current_colors = None
        self.current_position = None
        self.current_direction = None
        self.score = 0
        self.game_over = False
        self.trajectory = []

    def new_piece(self):
        piece_data = generate_random_shape()
        self.current_piece = piece_data["shape"]
        self.current_colors = piece_data["colors"]
        
        # Try all four directions
        directions = ['up', 'down', 'left', 'right']
        random.shuffle(directions)
        
        for direction in directions:
            if self.can_spawn_piece(direction):
                self.current_direction = direction
                if direction == 'up':
                    self.current_position = [0, GRID_SIZE // 2 - len(self.current_piece[0]) // 2]
                elif direction == 'down':
                    self.current_position = [GRID_SIZE - len(self.current_piece), GRID_SIZE // 2 - len(self.current_piece[0]) // 2]
                elif direction == 'left':
                    self.current_position = [GRID_SIZE // 2 - len(self.current_piece) // 2, 0]
                else:  # right
                    self.current_position = [GRID_SIZE // 2 - len(self.current_piece) // 2, GRID_SIZE - len(self.current_piece[0])]
                return True
        
        self.game_over = True
        return False

    def can_spawn_piece(self, direction):
        if direction == 'up':
            return all(self.grid[0][j] == 0 for j in range(GRID_SIZE))
        elif direction == 'down':
            return all(self.grid[GRID_SIZE-1][j] == 0 for j in range(GRID_SIZE))
        elif direction == 'left':
            return all(self.grid[i][0] == 0 for i in range(GRID_SIZE))
        else:  # right
            return all(self.grid[i][GRID_SIZE-1] == 0 for i in range(GRID_SIZE))

    def move(self, dx, dy):
        if self.current_piece is None or self.current_position is None:
            return False
        new_x = self.current_position[0] + dx
        new_y = self.current_position[1] + dy
        if self.is_valid_position(new_x, new_y):
            self.current_position = [new_x, new_y]
            return True
        return False

    def rotate(self):
        if self.current_piece is None:
            return
        rotated_piece = list(zip(*self.current_piece[::-1]))
        if self.is_valid_position(self.current_position[0], self.current_position[1], rotated_piece):
            self.current_piece = rotated_piece

    def is_valid_position(self, x, y, piece=None):
        if piece is None:
            if self.current_piece is None:
                return False
            piece = self.current_piece
        for i, row in enumerate(piece):
            for j, cell in enumerate(row):
                if cell:
                    if (x + i < 0 or x + i >= GRID_SIZE or
                        y + j < 0 or y + j >= GRID_SIZE or
                        self.grid[x + i][y + j] != 0):
                        return False
        return True

    def merge_piece(self):
        if self.current_piece is None or self.current_position is None:
            return
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    color_index = (i * len(self.current_piece[0]) + j) % len(self.current_colors)
                    self.grid[self.current_position[0] + i][self.current_position[1] + j] = self.current_colors[color_index]

    def check_lines(self):
        lines_cleared = 0
        for i in range(GRID_SIZE):
            if all(self.grid[i]):
                del self.grid[i]
                self.grid.insert(0, [0 for _ in range(GRID_SIZE)])
                lines_cleared += 1
        return lines_cleared

    def check_columns(self):
        columns_cleared = 0
        for j in range(GRID_SIZE):
            if all(self.grid[i][j] for i in range(GRID_SIZE)):
                for i in range(GRID_SIZE):
                    self.grid[i][j] = 0
                columns_cleared += 1
        return columns_cleared

    def check_color_match(self):
        matches = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] != 0:
                    color = self.grid[i][j]
                    if (i > 0 and self.grid[i-1][j] == color) or \
                       (i < GRID_SIZE-1 and self.grid[i+1][j] == color) or \
                       (j > 0 and self.grid[i][j-1] == color) or \
                       (j < GRID_SIZE-1 and self.grid[i][j+1] == color):
                        matches += 1
                        self.grid[i][j] = 0
        return matches

    def update(self):
        if self.current_piece is None:
            if not self.new_piece():
                return

        if self.current_direction == 'up':
            moved = self.move(1, 0)
        elif self.current_direction == 'down':
            moved = self.move(-1, 0)
        elif self.current_direction == 'left':
            moved = self.move(0, 1)
        else:  # right
            moved = self.move(0, -1)

        if not moved:
            self.merge_piece()
            lines_cleared = self.check_lines()
            columns_cleared = self.check_columns()
            color_matches = self.check_color_match()
            self.score += (lines_cleared + columns_cleared) * 100 + color_matches * 50
            self.current_piece = None

    def draw(self, screen):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pygame.draw.rect(screen, COLOR_MAP[self.grid[i][j]], (j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, WHITE, (j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        if self.current_piece and self.current_position:
            for i, row in enumerate(self.current_piece):
                for j, cell in enumerate(row):
                    if cell:
                        color_index = (i * len(self.current_piece[0]) + j) % len(self.current_colors)
                        pygame.draw.rect(screen, COLOR_MAP[self.current_colors[color_index]],
                                         ((self.current_position[1] + j) * CELL_SIZE,
                                          (self.current_position[0] + i) * CELL_SIZE,
                                          CELL_SIZE, CELL_SIZE))

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))

# The main function and game loop remain the same
