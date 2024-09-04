import random
import pygame
from typing import List, Tuple, FrozenSet
from arc_dsl.dsl import *  # Import all DSL functions

# Constants
CELL_SIZE = 20
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

class Tetris:
    def __init__(self, grid_size=30):
        self.grid_size = grid_size
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_piece = None
        self.current_colors = None
        self.current_position = None
        self.score = 0
        self.game_over = False

    def generate_random_shape(self):
        shape_type = random.choice(["tetromino", "complex", "pattern"])
        
        if shape_type == "tetromino":
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
        elif shape_type == "complex":
            size = random.randint(3, 5)
            shape = [[random.randint(0, 1) for _ in range(size)] for _ in range(size)]
        else:  # pattern
            patterns = [
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]],  # X pattern
                [[1, 1, 1], [1, 0, 1], [1, 1, 1]],  # O pattern
                [[1, 1, 1], [0, 0, 0], [1, 1, 1]],  # = pattern
            ]
            shape = random.choice(patterns)
        
        colors = [random.randint(1, 9) for _ in range(random.randint(1, 3))]
        return {"shape": shape, "colors": colors}

    def new_piece(self):
        piece_data = self.generate_random_shape()
        self.current_piece = piece_data["shape"]
        self.current_colors = piece_data["colors"]
        self.current_position = [0, self.grid_size // 2 - len(self.current_piece[0]) // 2]
        
        if not self.is_valid_position(self.current_position[0], self.current_position[1]):
            self.game_over = True
            return False
        return True

    def move(self, dx, dy):
        if self.current_piece is None or self.current_position is None:
            return False
        new_position = add((self.current_position[0], self.current_position[1]), (dx, dy))
        if self.is_valid_position(new_position[0], new_position[1]):
            self.current_position = list(new_position)
            return True
        return False

    def rotate(self):
        if self.current_piece is None:
            return
        rotated_piece = rot90(self.current_piece)
        if self.is_valid_position(self.current_position[0], self.current_position[1], rotated_piece):
            self.current_piece = rotated_piece

    def is_valid_position(self, x, y, piece=None):
        if piece is None:
            piece = self.current_piece
        for i, row in enumerate(piece):
            for j, cell in enumerate(row):
                if cell:
                    grid_pos = add((x, y), (i, j))
                    if not (0 <= grid_pos[0] < self.grid_size and 0 <= grid_pos[1] < self.grid_size) or \
                       self.grid[grid_pos[0]][grid_pos[1]] != 0:
                        return False
        return True

    def merge_piece(self):
        if self.current_piece is None or self.current_position is None:
            return
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    grid_pos = add((self.current_position[0], self.current_position[1]), (i, j))
                    color_index = (i * len(self.current_piece[0]) + j) % len(self.current_colors)
                    self.grid[grid_pos[0]][grid_pos[1]] = self.current_colors[color_index]

    def update(self):
        if self.current_piece is None:
            if not self.new_piece():
                return

        if not self.move(1, 0):
            self.merge_piece()
            self.clear_lines()
            self.apply_dsl_transformation()
            self.current_piece = None

    def clear_lines(self):
        lines_to_clear = [i for i in range(self.grid_size) if all(self.grid[i])]
        for line in lines_to_clear:
            del self.grid[line]
            self.grid.insert(0, [0 for _ in range(self.grid_size)])
        self.score += len(lines_to_clear) * 100

    def apply_dsl_transformation(self):
        transformation = random.choice(["rotate", "mirror", "color_change", "scale", "shift", "invert"])
        
        if transformation == "rotate":
            self.rotate_grid()
        elif transformation == "mirror":
            self.mirror_grid()
        elif transformation == "color_change":
            self.change_colors()
        elif transformation == "scale":
            self.scale_grid()
        elif transformation == "shift":
            self.shift_grid()
        elif transformation == "invert":
            self.invert_grid()

    def rotate_grid(self):
        self.grid = rot90(self.grid)

    def mirror_grid(self):
        axis = random.choice(["horizontal", "vertical"])
        if axis == "horizontal":
            self.grid = hmirror(self.grid)
        else:
            self.grid = vmirror(self.grid)

    def change_colors(self):
        color_map = {i: random.randint(1, 9) for i in range(1, 10)}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] != 0:
                    self.grid[i][j] = color_map[self.grid[i][j]]

    def scale_grid(self):
        scale_factor = random.choice([0.5, 2])
        new_size = int(self.grid_size * scale_factor)
        if scale_factor > 1:
            self.grid = upscale(self.grid, int(scale_factor))
        else:
            self.grid = downscale(self.grid, int(1/scale_factor))
        self.grid_size = new_size

    def shift_grid(self):
        direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.grid = shift(self.grid, direction)

    def invert_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] != 0:
                    self.grid[i][j] = invert(self.grid[i][j])

    def draw(self, screen):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pygame.draw.rect(screen, COLOR_MAP[self.grid[i][j]], (j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, WHITE, (j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        if self.current_piece and self.current_position:
            for i, row in enumerate(self.current_piece):
                for j, cell in enumerate(row):
                    if cell:
                        color_index = (i * len(self.current_piece[0]) + j) % len(self.current_colors)
                        grid_pos = add((self.current_position[0], self.current_position[1]), (i, j))
                        pygame.draw.rect(screen, COLOR_MAP[self.current_colors[color_index]],
                                         (grid_pos[1] * CELL_SIZE, grid_pos[0] * CELL_SIZE,
                                          CELL_SIZE, CELL_SIZE))

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
