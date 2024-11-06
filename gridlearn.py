from typing import Tuple, Set, FrozenSet, List, Dict, Any, Callable, Union
from random import choice, randint, sample, shuffle, uniform
import os
import hashlib
import importlib.util

from re_arc.dsl import *

# Define type aliases for clarity
Grid = Tuple[Tuple[int, ...], ...]
Indices = FrozenSet[Tuple[int, int]]
Patch = FrozenSet[Tuple[int, int]]
Object = FrozenSet[Tuple[int, Tuple[int, int]]]
Objects = FrozenSet[Object]
Numerical = Union[int, Tuple[int, int]]
Element = Union[Object, Grid]
Piece = Union[Grid, Patch]

# Path configurations
FINAL_FUNC_DEFS_DIR = 'final_func_defs'

def compute_grid_hash(grid: Grid) -> str:
    """
    Compute a unique hash for the grid based on its content.
    """
    grid_str = ''.join(''.join(map(str, row)) for row in grid)
    return hashlib.md5(grid_str.encode()).hexdigest()[:8]

def flatten_grid(grid: Grid) -> List[int]:
    """
    Flatten the grid into a single list of integers.
    """
    return [cell for row in grid for cell in row]

def analyze_grid(grid: Grid) -> Dict[str, Any]:
    """
    Analyze the grid to extract dimensions, palette, background color, and objects.
    """
    analysis = {}
    analysis['height'] = len(grid)
    analysis['width'] = len(grid[0]) if analysis['height'] > 0 else 0
    flattened = flatten_grid(grid)
    analysis['palette'] = palette(grid)
    analysis['bg_color'] = mostcommon(flattened)
    
    # Identify objects: connected components excluding background
    analysis['objects'] = objects(grid, univalued=True, diagonal=False, without_bg=True)
    
    return analysis

def detect_shapes(obj: Object) -> List[str]:
    """
    Detect shapes within an object and return a list of applicable DSL functions.
    Currently, it identifies if an object is a square, horizontal line, vertical line, etc.
    """
    shapes = []
    
    if len(obj) == 0:
        return shapes
    
    # Extract all positions
    positions = [pos for _, pos in obj]
    rows = [i for i, _ in positions]
    cols = [j for _, j in positions]
    
    unique_rows = set(rows)
    unique_cols = set(cols)
    
    # Check if object is a single cell
    if len(obj) == 1:
        shapes.append('fill_single')
        return shapes
    
    # Check for vertical line (all columns are the same)
    if len(unique_cols) == 1:
        shapes.append('vline')
    
    # Check for horizontal line (all rows are the same)
    if len(unique_rows) == 1:
        shapes.append('hline')
    
    # Check for square
    upper_left = ulcorner(obj)
    lower_right = lrcorner(obj)
    height = lower_right[0] - upper_left[0] + 1
    width = lower_right[1] - upper_left[1] + 1
    if height == width and len(obj) == height * width:
        shapes.append('square')
    else:
        # Additional checks for rotated squares or other shapes can be added here
        shapes.append('fill_complex')
    
    return shapes

def generate_dsl_commands(analysis: Dict[str, Any], transformations: List[str]=[]) -> List[str]:
    """
    Generate DSL commands based on the analyzed grid.
    Optionally apply transformations like rotations or mirrors.
    """
    commands = []
    
    height = analysis['height']
    width = analysis['width']
    bg_color = analysis['bg_color']
    
    # Initialize canvas
    commands.append(f"    gi = canvas({bg_color}, ({height}, {width}))")
    
    # Apply transformations before filling objects
    for transform in transformations:
        commands.append(f"    gi = {transform}(gi)")
    
    # Process each object
    for idx, obj in enumerate(analysis['objects'], start=1):
        obj_color = color(obj)
        obj_indices = toindices(obj)
        obj_indices_str = f"frozenset({set(obj_indices)})"
        
        # Detect shapes
        shapes = detect_shapes(obj)
        
        # Depending on the shape, choose the appropriate DSL function
        if 'vline' in shapes:
            commands.append(f"    gi = vline(gi, {obj_color}, {obj_indices_str})")
        elif 'hline' in shapes:
            commands.append(f"    gi = hline(gi, {obj_color}, {obj_indices_str})")
        elif 'square' in shapes:
            commands.append(f"    gi = fill(gi, {obj_color}, {obj_indices_str})")
        elif 'fill_complex' in shapes:
            commands.append(f"    gi = fill(gi, {obj_color}, {obj_indices_str})")
        elif 'fill_single' in shapes:
            commands.append(f"    gi = fill(gi, {obj_color}, {obj_indices_str})")
        else:
            # Default to fill for any other shapes
            commands.append(f"    gi = fill(gi, {obj_color}, {obj_indices_str})")
    
    # Correctly return the grid without passing any arguments
    commands.append("    return gi")
    return commands

def assemble_dsl_program(dsl_commands: List[str], grid_hash: str) -> str:
    """
    Assemble the DSL program with imports and the generator function.
    """
    import_line = "from re_arc.dsl import *"
    
    program_lines = [
        import_line,
        "",
        f"def generate_grid() -> Grid:",
        *dsl_commands
    ]
    return "\n".join(program_lines)

def save_dsl_program(program_content: str, grid_hash: str) -> str:
    """
    Save the DSL program to a file in the final_func_defs directory.
    """
    # Ensure the output directory exists
    os.makedirs(FINAL_FUNC_DEFS_DIR, exist_ok=True)
    
    # Define the filename with the hash
    filename = f"generate_{grid_hash}.py"
    filepath = os.path.join(FINAL_FUNC_DEFS_DIR, filename)
    
    # Write the DSL program to a file
    with open(filepath, 'w') as file:
        file.write(program_content)
    
    print(f"DSL program generated and saved to {filepath}")
    return filepath

def generate_dsl_program(grid: Grid, transformations: List[str]=[]) -> str:
    """
    Orchestrate the DSL program generation process.
    """
    # Analyze the grid
    analysis = analyze_grid(grid)
    
    # Compute a unique hash for the grid
    grid_hash = compute_grid_hash(grid)
    
    # Generate DSL commands
    dsl_commands = generate_dsl_commands(analysis, transformations=transformations)
    
    # Assemble the DSL program
    program_content = assemble_dsl_program(dsl_commands, grid_hash)
    
    # Save the DSL program
    filepath = save_dsl_program(program_content, grid_hash)
    
    return filepath

def execute_generated_dsl(filepath: str) -> Grid:
    """
    Dynamically import and execute the generated DSL function.
    """
    spec = importlib.util.spec_from_file_location("generated_module", filepath)
    generated_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated_module)
    function_name = "generate_grid"
    if not hasattr(generated_module, function_name):
        raise AttributeError(f"module '{generated_module.__name__}' has no attribute '{function_name}'")
    generate_func = getattr(generated_module, function_name)
    return generate_func()

def grids_match(grid1: Grid, grid2: Grid) -> bool:
    """
    Check if two grids are identical.
    """
    if len(grid1) != len(grid2):
        return False
    for row1, row2 in zip(grid1, grid2):
        if row1 != row2:
            return False
    return True

def learn_dsl_program(grid: Grid, max_attempts: int = 16) -> str:
    """
    Implement a simple learning mechanism that tries different transformations
    until the generated DSL program matches the target grid.
    """
    # List of possible transformations to try
    transformations_list = [
        [],
        ['rot90'],
        ['rot180'],
        ['rot270'],
        ['hmirror'],
        ['vmirror'],
        ['dmirror'],
        ['cmirror'],
        ['rot90', 'hmirror'],
        ['rot90', 'vmirror'],
        ['rot180', 'hmirror'],
        ['rot180', 'vmirror'],
        ['rot270', 'hmirror'],
        ['rot270', 'vmirror'],
        ['hmirror', 'vmirror'],
        ['vmirror', 'hmirror']
    ]
    
    attempt = 0
    for transforms in transformations_list:
        attempt += 1
        print(f"Attempt {attempt}: Applying transformations {transforms}")
        
        # Generate DSL program with current transformations
        generated_filepath = generate_dsl_program(grid, transformations=transforms)
        
        # Execute the generated DSL program
        try:
            generated_grid = execute_generated_dsl(generated_filepath)
        except Exception as e:
            print(f"Error executing DSL program: {e}")
            continue
        
        # Compare the generated grid with the target grid
        if grids_match(grid, generated_grid):
            print(f"Success: The generated DSL program {generated_filepath} correctly recreates the original grid.")
            return generated_filepath
        else:
            print(f"Mismatch detected for DSL program {generated_filepath}.")
    
    print("Error: Failed to generate a matching DSL program within the maximum number of attempts.")
    return ""

# Example Usage
if __name__ == "__main__":
    # Define your target grid here
    sample_grid = (
        (1, 1, 1, 1, 1, 1, 1),
        (1, 2, 2, 2, 2, 2, 1),
        (1, 2, 3, 3, 3, 2, 1),
        (1, 2, 3, 4, 3, 2, 1),
        (1, 2, 3, 3, 3, 2, 1),
        (1, 2, 2, 2, 2, 2, 1),
        (1, 1, 1, 1, 1, 1, 1),
    )
    
    # Attempt to learn the DSL program
    generated_filepath = learn_dsl_program(sample_grid)
    
    if generated_filepath:
        # Execute and verify
        try:
            generated_grid = execute_generated_dsl(generated_filepath)
        except Exception as e:
            print(f"Error during execution of generated DSL program: {e}")
            exit(1)
        
        # Final comparison
        if grids_match(sample_grid, generated_grid):
            print("Final Verification: The generated DSL program correctly recreates the original grid.")
        else:
            print("Final Verification: The generated DSL program does not match the original grid.")

