import os
import json
import hashlib
from typing import Tuple, Set, FrozenSet, List, Dict, Any
from collections import defaultdict, deque

# Importing DSL functions from re_arc.dsl
# Ensure that the 're_arc.dsl' module is in your Python path
from re_arc.dsl import *

Grid = Tuple[Tuple[int, ...], ...]
Indices = FrozenSet[Tuple[int, int]]
Patch = FrozenSet[Tuple[int, int]]
Object = FrozenSet[Tuple[int, Tuple[int, int]]]

# Path configurations
DSL_CONTEXT_PATH = os.path.join('data', 'dsl_context.json')
FINAL_FUNC_DEFS_DIR = 'final_func_defs'

def load_dsl_context(filepath: str) -> List[Dict[str, Any]]:
    """
    Load the DSL function definitions from a JSON file.
    """
    with open(filepath, 'r') as file:
        dsl_context = json.load(file)
    return dsl_context

def compute_grid_hash(grid: Grid) -> str:
    """
    Compute a unique hash for the grid based on its content.
    """
    grid_str = ''.join(''.join(map(str, row)) for row in grid)
    return hashlib.md5(grid_str.encode()).hexdigest()[:8]

def analyze_grid(grid: Grid) -> Dict[str, Any]:
    """
    Analyze the grid to extract dimensions, palette, background color, and objects.
    """
    analysis = {}
    analysis['height'] = len(grid)
    analysis['width'] = len(grid[0]) if analysis['height'] > 0 else 0
    analysis['palette'] = palette(grid)
    analysis['bg_color'] = mostcommon(grid)
    
    # Identify objects: connected components excluding background
    analysis['objects'] = objects(grid, univalued=True, diagonal=False, without_bg=True)
    
    return analysis

def generate_dsl_commands(analysis: Dict[str, Any]) -> List[str]:
    """
    Generate DSL commands based on the analyzed grid.
    """
    commands = []
    
    height = analysis['height']
    width = analysis['width']
    bg_color = analysis['bg_color']
    
    # Initialize canvas
    commands.append(f"    gi = canvas({bg_color}, ({height}, {width}))")
    
    # Process each object
    for idx, obj in enumerate(analysis['objects'], start=1):
        obj_color = color(obj)
        obj_indices = toindices(obj)
        obj_indices_str = f"frozenset({set(obj_indices)})"
        
        # Fill the object with its color
        commands.append(f"    gi = fill(gi, {obj_color}, {obj_indices_str})")
    
    commands.append("    return gi")
    return commands

def assemble_dsl_program(dsl_commands: List[str], grid_hash: str) -> str:
    """
    Assemble the DSL program with imports and the generator function.
    """
    program_lines = [
        "from re_arc.dsl import *",
        "",
        f"def generate_{grid_hash}() -> Grid:",
        *dsl_commands
    ]
    return "\n".join(program_lines)

def save_dsl_program(program_content: str, grid_hash: str) -> str:
    """
    Save the DSL program to a file in the final_func_defs directory.
    """
    # Ensure the output directory exists
    os.makedirs(FINAL_FUNC_DEFS_DIR, exist_ok=True)
    
    # Define the filename
    filename = f"generate_{grid_hash}.py"
    filepath = os.path.join(FINAL_FUNC_DEFS_DIR, filename)
    
    # Write the DSL program to a file
    with open(filepath, 'w') as file:
        file.write(program_content)
    
    print(f"DSL program generated and saved to {filepath}")
    return filepath

def generate_dsl_program(grid: Grid) -> str:
    """
    Orchestrate the DSL program generation process.
    
    Args:
        grid (Grid): The target grid to reproduce.
        
    Returns:
        str: The filepath of the generated DSL program.
    """
    # Load DSL context (not used in this simplified version but available for extensions)
    # dsl_context = load_dsl_context(DSL_CONTEXT_PATH)
    
    # Analyze the grid
    analysis = analyze_grid(grid)
    
    # Compute a unique hash for the grid
    grid_hash = compute_grid_hash(grid)
    
    # Generate DSL commands
    dsl_commands = generate_dsl_commands(analysis)
    
    # Assemble the DSL program
    program_content = assemble_dsl_program(dsl_commands, grid_hash)
    
    # Save the DSL program
    filepath = save_dsl_program(program_content, grid_hash)
    
    return filepath

# Helper Functions for Connected Components (if needed)
# The 'objects' function from the DSL is used, but you might need additional helpers based on DSL capabilities.

# Example Usage
if __name__ == "__main__":
    # Example grid
    sample_grid = (
        (1, 1, 1, 1, 1),
        (1, 2, 2, 2, 1),
        (1, 2, 3, 2, 1),
        (1, 2, 2, 2, 1),
        (1, 1, 1, 1, 1),
    )
    
    # Generate the DSL program
    generate_dsl_program(sample_grid)

