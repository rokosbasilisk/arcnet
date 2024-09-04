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
    shape_type = random.choice(["tetromino", "complex", "pattern"])
    
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
    return {"shape": [list(row) for row in shape], "colors": [color]}

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
        self.current_pieces = []
        self.score = 0
        self.game_over = False

    def new_piece(self):
        piece_data = generate_random_shape()
        new_piece = {
            "shape": piece_data["shape"],
            "colors": piece_data["colors"],
            "position": [0, GRID_SIZE // 2 - len(piece_data["shape"][0]) // 2],
            "rotation": 0,
            "velocity": [1, 0]  # [dy, dx]
        }
        self.current_pieces.append(new_piece)
        
        if not self.is_valid_position(new_piece):
            self.game_over = True
            return False
        return True

    def move(self, piece):
        new_position = [
            piece["position"][0] + piece["velocity"][0],
            piece["position"][1] + piece["velocity"][1]
        ]
        if self.is_valid_position(piece, new_position):
            piece["position"] = new_position
            return True
        return False

    def rotate(self, piece):
        rotated_shape = [list(row) for row in zip(*piece["shape"][::-1])]
        if self.is_valid_position(piece, piece["position"], rotated_shape):
            piece["shape"] = rotated_shape
            piece["rotation"] = (piece["rotation"] + 90) % 360

    def is_valid_position(self, piece, position=None, shape=None):
        if position is None:
            position = piece["position"]
        if shape is None:
            shape = piece["shape"]
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    x, y = position[0] + i, position[1] + j
                    if (x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE or
                        self.grid[x][y] != 0):
                        return False
        return True

    def merge_piece(self, piece):
        for i, row in enumerate(piece["shape"]):
            for j, cell in enumerate(row):
                if cell:
                    color_index = (i * len(piece["shape"][0]) + j) % len(piece["colors"])
                    self.grid[piece["position"][0] + i][piece["position"][1] + j] = piece["colors"][color_index]

    def update(self):
        if not self.current_pieces:
            self.new_piece()
            if random.random() < 0.3:  # 30% chance for multiple pieces
                self.new_piece()

        for piece in self.current_pieces[:]:
            if not self.move(piece):
                self.merge_piece(piece)
                self.current_pieces.remove(piece)
                self.apply_random_transformation()

        self.clear_lines()

        if not self.current_pieces:
            self.new_piece()

    def clear_lines(self):
        lines_to_clear = [i for i in range(GRID_SIZE) if all(self.grid[i])]
        for line in lines_to_clear:
            del self.grid[line]
            self.grid.insert(0, [0 for _ in range(GRID_SIZE)])
        self.score += len(lines_to_clear) * 100

    def apply_random_transformation(self):
        transformations = [
            self.flood_fill,
            self.shapeshift,
            self.multiply,
            self.color_shift,
            self.shear,
            self.displacement,
            self.rotate_grid
        ]
        random.choice(transformations)()

    def flood_fill(self):
        start_i, start_j = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        color = random.randint(1, 9)
        target_color = self.grid[start_i][start_j]
        self._flood_fill_recursive(start_i, start_j, target_color, color)

    def _flood_fill_recursive(self, i, j, target_color, replacement_color):
        if (i < 0 or i >= GRID_SIZE or j < 0 or j >= GRID_SIZE or
            self.grid[i][j] != target_color):
            return
        self.grid[i][j] = replacement_color
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            self._flood_fill_recursive(i + di, j + dj, target_color, replacement_color)

    def shapeshift(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] != 0 and random.random() < 0.1:
                    new_shape = generate_random_shape()
                    self._place_shape(i, j, new_shape["shape"], new_shape["colors"])

    def _place_shape(self, i, j, shape, colors):
        for di, row in enumerate(shape):
            for dj, cell in enumerate(row):
                if cell and 0 <= i + di < GRID_SIZE and 0 <= j + dj < GRID_SIZE:
                    color_index = (di * len(shape[0]) + dj) % len(colors)
                    self.grid[i + di][j + dj] = colors[color_index]

    def multiply(self):
        new_pieces = []
        for piece in self.current_pieces:
            if random.random() < 0.3:  # 30% chance to multiply
                new_piece = copy.deepcopy(piece)
                new_piece["position"] = [piece["position"][0], piece["position"][1] + len(piece["shape"][0])]
                if self.is_valid_position(new_piece):
                    new_pieces.append(new_piece)
        self.current_pieces.extend(new_pieces)

    def color_shift(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] != 0:
                    self.grid[i][j] = (self.grid[i][j] % 9) + 1

    def shear(self):
        shear_factor = random.uniform(-0.5, 0.5)
        new_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                new_j = int(j + i * shear_factor) % GRID_SIZE
                new_grid[i][new_j] = self.grid[i][j]
        self.grid = new_grid

    def displacement(self):
        displacement = [random.randint(-2, 2), random.randint(-2, 2)]
        new_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                new_i = (i + displacement[0]) % GRID_SIZE
                new_j = (j + displacement[1]) % GRID_SIZE
                new_grid[new_i][new_j] = self.grid[i][j]
        self.grid = new_grid

    def rotate_grid(self):
        self.grid = [list(row) for row in zip(*self.grid[::-1])]

    def draw(self, screen):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pygame.draw.rect(screen, COLOR_MAP[self.grid[i][j]], (j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, WHITE, (j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        for piece in self.current_pieces:
            for i, row in enumerate(piece["shape"]):
                for j, cell in enumerate(row):
                    if cell:
                        color_index = (i * len(piece["shape"][0]) + j) % len(piece["colors"])
                        pygame.draw.rect(screen, COLOR_MAP[piece["colors"][color_index]],
                                         ((piece["position"][1] + j) * CELL_SIZE,
                                          (piece["position"][0] + i) * CELL_SIZE,
                                          CELL_SIZE, CELL_SIZE))

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
