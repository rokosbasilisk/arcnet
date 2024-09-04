import pygame
import random
import sys
import json
import copy

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

def generate_random_shape():
    shape_type = random.choice([
        "tetromino", "complex", "pattern"
    ])
    
    if shape_type == "tetromino":
        return generate_tetromino()
    elif shape_type == "complex":
        return generate_complex_shape()
    else:
        return generate_pattern()

def generate_tetromino():
    shapes = [
        [[1, 1, 1, 1]],  # I
        [[1, 1], [1, 1]],  # O
        [[1, 1, 1], [0, 1, 0]],  # T
        [[1, 1, 1], [1, 0, 0]],  # L
        [[1, 1, 1], [0, 0, 1]],  # J
        [[1, 1, 0], [0, 1, 1]],  # S
        [[0, 1, 1], [1, 1, 0]]   # Z
    ]
    shape = random.choice(shapes)
    color = random.randint(1, 9)
    return {"shape": shape, "colors": [color]}

def generate_complex_shape():
    size = random.randint(3, 5)
    shape = [[random.randint(0, 1) for _ in range(size)] for _ in range(size)]
    colors = [random.randint(1, 9) for _ in range(random.randint(1, 3))]
    return {"shape": shape, "colors": colors}

def generate_pattern():
    patterns = [
        {"shape": [[1, 0, 1], [0, 1, 0], [1, 0, 1]], "colors": [random.randint(1, 9)]},  # X pattern
        {"shape": [[1, 1, 1], [1, 0, 1], [1, 1, 1]], "colors": [random.randint(1, 9)]},  # O pattern
        {"shape": [[1, 1, 1], [0, 0, 0], [1, 1, 1]], "colors": [random.randint(1, 9), random.randint(1, 9)]},  # = pattern
    ]
    return random.choice(patterns)

class Tetris:
    def __init__(self):
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.current_piece = None
        self.current_colors = None
        self.current_position = None
        self.score = 0
        self.game_over = False

    def new_piece(self):
        piece_data = generate_random_shape()
        self.current_piece = piece_data["shape"]
        self.current_colors = piece_data["colors"]
        self.current_position = [0, GRID_SIZE // 2 - len(self.current_piece[0]) // 2]
        
        if not self.is_valid_position(self.current_position[0], self.current_position[1]):
            self.game_over = True
            return False
        return True

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

    def update(self):
        if self.current_piece is None:
            if not self.new_piece():
                return

        if not self.move(1, 0):
            self.merge_piece()
            self.clear_lines()
            self.apply_random_transformation()
            self.current_piece = None

    def clear_lines(self):
        lines_to_clear = [i for i in range(GRID_SIZE) if all(self.grid[i])]
        for line in lines_to_clear:
            del self.grid[line]
            self.grid.insert(0, [0 for _ in range(GRID_SIZE)])
        self.score += len(lines_to_clear) * 100

    def apply_random_transformation(self):
        transformation = random.choice(["multiply", "flood_fill", "zoom", "color_change", "color_inversion"])
        
        if transformation == "multiply":
            self.multiply_objects()
        elif transformation == "flood_fill":
            self.flood_fill()
        elif transformation == "zoom":
            self.zoom()
        elif transformation == "color_change":
            self.color_change()
        elif transformation == "color_inversion":
            self.color_inversion()

    def multiply_objects(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] != 0 and random.random() < 0.1:  # 10% chance to multiply
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and self.grid[ni][nj] == 0:
                            self.grid[ni][nj] = self.grid[i][j]

    def flood_fill(self):
        start_i, start_j = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        color = random.randint(1, 9)
        self._flood_fill_recursive(start_i, start_j, self.grid[start_i][start_j], color)

    def _flood_fill_recursive(self, i, j, target_color, replacement_color):
        if i < 0 or i >= GRID_SIZE or j < 0 or j >= GRID_SIZE:
            return
        if self.grid[i][j] != target_color:
            return
        self.grid[i][j] = replacement_color
        self._flood_fill_recursive(i+1, j, target_color, replacement_color)
        self._flood_fill_recursive(i-1, j, target_color, replacement_color)
        self._flood_fill_recursive(i, j+1, target_color, replacement_color)
        self._flood_fill_recursive(i, j-1, target_color, replacement_color)

    def zoom(self):
        zoom_in = random.choice([True, False])
        factor = 2 if zoom_in else 0.5
        center_i, center_j = GRID_SIZE // 2, GRID_SIZE // 2
        new_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                source_i = int((i - center_i) * factor + center_i)
                source_j = int((j - center_j) * factor + center_j)
                if 0 <= source_i < GRID_SIZE and 0 <= source_j < GRID_SIZE:
                    new_grid[i][j] = self.grid[source_i][source_j]
        
        self.grid = new_grid

    def color_change(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] != 0:
                    self.grid[i][j] = random.randint(1, 9)

    def color_inversion(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] != 0:
                    self.grid[i][j] = 10 - self.grid[i][j]  # Invert color (1 becomes 9, 2 becomes 8, etc.)

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
