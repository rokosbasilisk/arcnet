import os
import hashlib
import importlib.util
import inspect
import itertools
from typing import Tuple, Set, FrozenSet, List, Dict, Any, Union
from collections import deque
from tqdm import tqdm

from re_arc.dsl import *  # Import all existing DSL functions
import re_arc.dsl as dsl

# Define type aliases for clarity
Grid = Tuple[Tuple[int, ...], ...]
Indices = FrozenSet[Tuple[int, int]]
Object = FrozenSet[Tuple[int, Tuple[int, int]]]
Objects = FrozenSet[Object]
Element = Union[Object, Grid]

# Path configurations
FINAL_FUNC_DEFS_DIR = 'final_func_defs'

def compute_grid_hash(grid: Grid) -> str:
    """
    Compute a unique hash for the grid based on its content.
    """
    grid_str = ''.join(''.join(map(str, row)) for row in grid)
    return hashlib.md5(grid_str.encode()).hexdigest()[:8]

def grids_match(grid1: Grid, grid2: Grid) -> bool:
    """
    Check if two grids are identical.
    """
    return grid1 == grid2

def assemble_program(expression: str) -> str:
    """
    Assemble the DSL program with imports and the generator function.
    """
    import_line = "from re_arc.dsl import *"
    program_lines = [
        import_line,
        "",
        f"def generate_grid() -> Grid:",
        f"    return {expression}"
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

def synthesize_program(target_grid: Grid, max_depth: int = 5) -> Union[str, None]:
    """
    Synthesize a DSL program that generates the target grid using iterative deepening tree search.
    """
    grid_hash = compute_grid_hash(target_grid)
    
    # Analyze the grid to extract objects
    try:
        analyzed = dsl.objects(target_grid, univalued=True, diagonal=False, without_bg=True)
    except Exception as e:
        print(f"Error during grid analysis: {e}")
        return None
    
    objects = analyzed
    if not objects:
        print("No objects detected in the grid. Only a canvas may be needed.")
    
    # Start building the nested DSL expression
    # Begin with the canvas function
    bg_color = dsl.mostcommon(target_grid)
    height = len(target_grid)
    width = len(target_grid[0]) if height > 0 else 0
    expression = f"canvas({bg_color}, ({height}, {width}))"
    
    # Iterate over each object and apply the fill function
    # To optimize, sort objects by size (largest first)
    sorted_objects = sorted(objects, key=lambda obj: len(obj), reverse=True)
    
    for obj in sorted_objects:
        color = dsl.color(obj)
        patch = dsl.toindices(obj)
        patch_expr = f"frozenset({sorted(patch)})"
        expression = f"fill({expression}, {color}, {patch_expr})"
    
    # Assemble the DSL program
    program_content = assemble_program(expression)
    return program_content

def generate_dsl_program(grid: Grid, max_depth: int = 5) -> str:
    """
    Orchestrate the DSL program generation process.
    """
    program_content = synthesize_program(grid, max_depth)
    if program_content:
        grid_hash = compute_grid_hash(grid)
        filepath = save_dsl_program(program_content, grid_hash)
        return filepath
    else:
        print("Failed to synthesize a program that generates the grid.")
        return ""

def learn_dsl_program(grid: Grid) -> str:
    """
    Learn and generate a DSL program that recreates the given grid.
    """
    generated_filepath = generate_dsl_program(grid)
    if generated_filepath:
        # Execute and verify
        try:
            generated_grid = execute_generated_dsl(generated_filepath)
        except Exception as e:
            print(f"Error during execution of generated DSL program: {e}")
            return ""
    
        # Final comparison
        if grids_match(grid, generated_grid):
            print("Final Verification: The generated DSL program correctly recreates the original grid.")
            return generated_filepath
        else:
            print("Final Verification: The generated DSL program does not match the original grid.")
            return ""
    else:
        print("No DSL program was generated.")
        return ""

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

