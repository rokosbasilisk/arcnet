import os
import hashlib
import importlib.util
from typing import Tuple, Set, FrozenSet, List, Dict, Any, Union

from re_arc.dsl import *  # Import all existing DSL functions

# Define type aliases for clarity
Grid = Tuple[Tuple[int, ...], ...]
Indices = Set[Tuple[int, int]]
Object = Set[Tuple[int, Tuple[int, int]]]
Objects = Set[Object]
Numerical = Union[int, Tuple[int, int]]
Element = Union[Object, Grid]

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
    analysis['grid'] = grid
    analysis['flattened'] = flatten_grid(grid)
    
    try:
        analysis['palette'] = palette(grid)  # Using existing palette function
    except NameError:
        print("Error: 'palette' function is not defined in re_arc.dsl module.")
        analysis['palette'] = frozenset()
    
    try:
        analysis['bg_color'] = mostcommon(analysis['flattened'])  # Using existing mostcommon function
    except NameError:
        print("Error: 'mostcommon' function is not defined in re_arc.dsl module.")
        analysis['bg_color'] = 0  # Default background color
    
    try:
        analysis['objects'] = objects(grid, univalued=True, diagonal=False, without_bg=True)  # Using existing objects function
    except NameError:
        print("Error: 'objects' function is not defined in re_arc.dsl module.")
        analysis['objects'] = set()
    
    return analysis

def detect_patterns(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect high-level patterns in the grid and represent them using DSL functions.
    """
    patterns = []
    grid = analysis['grid']
    height = analysis['height']
    width = analysis['width']
    bg_color = analysis['bg_color']

    # Detect Borders
    border_indices = set()
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                border_indices.add((i, j))
    border_colors = set(grid[i][j] for i, j in border_indices)
    if len(border_colors) == 1 and bg_color not in border_colors:
        border_color = next(iter(border_colors))
        patterns.append({
            'type': 'border',
            'color': border_color
        })

    # Detect Filled Rectangles (excluding borders)
    filled_indices = {(i, j) for i in range(1, height-1) for j in range(1, width-1) if grid[i][j] != bg_color}
    if filled_indices:
        from collections import defaultdict
        color_indices = defaultdict(set)
        for i, j in filled_indices:
            color_indices[grid[i][j]].add((i, j))
        for color, indices in color_indices.items():
            min_i = min(i for i, _ in indices)
            max_i = max(i for i, _ in indices)
            min_j = min(j for _, j in indices)
            max_j = max(j for _, j in indices)
            expected_indices = {(i, j) for i in range(min_i, max_i + 1) for j in range(min_j, max_j + 1)}
            if indices == expected_indices:
                patterns.append({
                    'type': 'rectangle',
                    'color': color,
                    'position': (min_i, min_j),
                    'size': (max_i - min_i + 1, max_j - min_j + 1)
                })
            else:
                # Complex shapes; fallback to fill if they are not simple rectangles
                patterns.append({
                    'type': 'fill',
                    'color': color,
                    'indices': indices
                })

    return patterns

def generate_nested_expression(dsl_commands: List[str]) -> str:
    """
    Generate a single nested expression from a list of DSL commands.
    """
    nested_expr = dsl_commands[0].strip()
    for cmd in dsl_commands[1:]:
        # Each command is expected to be a 'fill' function call
        # Example: fill(canvas(...), 2, frozenset({...}))
        nested_expr = f"fill({nested_expr}, {cmd.split(',')[1].strip()}, {cmd.split(',')[2].strip()})"
    return nested_expr

def generate_dsl_commands_from_patterns(patterns: List[Dict[str, Any]], analysis: Dict[str, Any], transformations: List[str]) -> List[str]:
    """
    Generate DSL commands based on detected patterns using a greedy approach to optimize.
    """
    commands = []
    height = analysis['height']
    width = analysis['width']
    bg_color = analysis['bg_color']

    # Start with canvas
    commands.append(f"canvas({bg_color}, ({height}, {width}))")

    # Apply transformations before drawing (if any)
    for transform in transformations:
        commands[-1] = f"{transform}({commands[-1]})"

    for pattern in patterns:
        if pattern['type'] == 'border':
            # Since 'border' function is not defined, use 'fill' to manually fill the border
            border_color = pattern['color']
            border_indices = set()
            for i in range(height):
                for j in range(width):
                    if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                        border_indices.add((i, j))
            indices_str = f"frozenset({sorted(border_indices)})"
            commands.append(f"fill(gi, {border_color}, {indices_str})")
        elif pattern['type'] == 'rectangle':
            # Use 'fill' to fill the rectangle
            color = pattern['color']
            pos = pattern['position']
            size = pattern['size']
            min_i, min_j = pos
            rect_height, rect_width = size
            rect_indices = {(i, j) for i in range(min_i, min_i + rect_height) for j in range(min_j, min_j + rect_width)}
            indices_str = f"frozenset({sorted(rect_indices)})"
            commands.append(f"fill(gi, {color}, {indices_str})")
        elif pattern['type'] == 'fill':
            # Use 'fill' for complex shapes
            color = pattern['color']
            indices = pattern['indices']
            indices_str = f"frozenset({sorted(indices)})"
            commands.append(f"fill(gi, {color}, {indices_str})")
        # Add more pattern types and their corresponding DSL functions as needed

    return commands

def assemble_dsl_program(nested_expression: str, grid_hash: str) -> str:
    """
    Assemble the DSL program with imports and the generator function.
    """
    import_line = "from re_arc.dsl import *"  # Import all DSL functions

    program_lines = [
        import_line,
        "",
        f"def generate_grid() -> Grid:",
        f"    return {nested_expression}"
    ]
    return "\n".join(program_lines)

def save_dsl_program(program_content: str, grid_hash: str) -> str:
    """
    Save the DSL program to a file in the final_func_defs directory.
    """
    os.makedirs(FINAL_FUNC_DEFS_DIR, exist_ok=True)
    filename = f"generate_{grid_hash}.py"
    filepath = os.path.join(FINAL_FUNC_DEFS_DIR, filename)
    with open(filepath, 'w') as file:
        file.write(program_content)
    print(f"DSL program generated and saved to {filepath}")
    return filepath

def generate_dsl_program(grid: Grid, transformations: List[str]=[]) -> str:
    """
    Orchestrate the DSL program generation process.
    """
    analysis = analyze_grid(grid)
    grid_hash = compute_grid_hash(grid)
    patterns = detect_patterns(analysis)
    dsl_commands = generate_dsl_commands_from_patterns(patterns, analysis, transformations)
    nested_expression = generate_nested_expression(dsl_commands)
    program_content = assemble_dsl_program(nested_expression, grid_hash)
    filepath = save_dsl_program(program_content, grid_hash)
    return filepath

def execute_generated_dsl(filepath: str) -> Grid:
    """
    Dynamically import and execute the generated DSL function.
    """
    spec = importlib.util.spec_from_file_location("generated_module", filepath)
    generated_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(generated_module)
    except Exception as e:
        raise ImportError(f"Failed to load DSL module from {filepath}: {e}")
    
    function_name = "generate_grid"
    if not hasattr(generated_module, function_name):
        raise AttributeError(f"module '{generated_module.__name__}' has no attribute '{function_name}'")
    
    generate_func = getattr(generated_module, function_name)
    return generate_func()

def grids_match(grid1: Grid, grid2: Grid) -> bool:
    """
    Check if two grids are identical.
    """
    return grid1 == grid2

def learn_dsl_program(grid: Grid) -> str:
    """
    Attempt to learn a DSL program that generates the grid using pattern detection.
    """
    patterns = detect_patterns(analyze_grid(grid))
    # Sort patterns by the number of cells they cover (descending)
    patterns_sorted = sorted(patterns, key=lambda p: len(p.get('indices', [])) if p['type'] == 'fill' else float('inf'), reverse=True)
    # Generate DSL commands based on sorted patterns
    dsl_commands = generate_dsl_commands_from_patterns(patterns_sorted, analyze_grid(grid), transformations=[])
    # Generate nested expression
    nested_expression = "canvas({0}, {1})".format(
        grid[0][0],  # Assuming all cells are initially bg_color
        (len(grid), len(grid[0]))
    )
    for cmd in dsl_commands[1:]:  # Skip the first 'canvas' command
        # Each 'cmd' is a 'fill' function call like 'fill(gi, 2, frozenset({...}))'
        # Extract color and indices
        parts = cmd.split(',', 2)
        if len(parts) < 3:
            continue  # Skip invalid commands
        color = parts[1].strip()
        indices = parts[2].strip().rstrip(')')
        nested_expression = f"fill({nested_expression}, {color}, {indices})"
    
    # Assemble the DSL program
    grid_hash = compute_grid_hash(grid)
    program_content = assemble_dsl_program(nested_expression, grid_hash)
    # Save the DSL program
    filepath = save_dsl_program(program_content, grid_hash)
    # Execute and verify
    try:
        generated_grid = execute_generated_dsl(filepath)
    except Exception as e:
        print(f"Error during execution of generated DSL program: {e}")
        return ""
    if grids_match(grid, generated_grid):
        print(f"Success: The generated DSL program {filepath} correctly recreates the original grid.")
        return filepath
    else:
        print(f"Mismatch detected for DSL program {filepath}.")
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

