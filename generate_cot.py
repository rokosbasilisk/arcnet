import json
import random
import re
import os
from typing import List
import inspect
from transform_functions import *  # Import all transformation functions
from dsl import *  # Import all DSL functions


# Load the functions context to extract functions that return a "Grid"
with open('data/functions_context.json', 'r') as f:
    functions_context = json.load(f)

# Extract the list of function names where return_type is "Grid"
action_list = [func['name'] for func in functions_context if func['return_type'] == 'Grid']

def compress_grid(grid):
    """Compress a grid into a run-length encoded string representation."""
    if not grid or not grid[0]:
        return ""
    
    # Flatten the grid into a list of strings
    flattened = [str(cell) for row in grid for cell in row]
    compressed = []
    current_char = flattened[0]
    count = 1
    
    # Iterate through the flattened list to create a run-length encoding
    for char in flattened[1:]:
        if char == current_char:
            count += 1
        else:
            compressed.append(f"{current_char}{count}")
            current_char = char
            count = 1
    
    # Add the final run-length encoded segment
    compressed.append(f"{current_char}{count}")
    return "".join(compressed)

# Load the dataset of tasks
with open('data/arc-agi_training_challenges.json', 'r') as f:
    tasks_data = json.load(f)

# Function to extract the transform logic from the transformation functions
def get_transform_function_source(hash_id: str) -> str:
    """Returns the source code of the transform function as a string."""
    func_name = f'verify_{hash_id}'
    if func_name in globals():
        func_source = inspect.getsource(globals()[func_name])
        return func_source
    else:
        raise ValueError(f"No function found for hash ID: {hash_id}")

# Parse the function source to detect action_list calls
def parse_action_calls(func_source: str, action_list: List[str]) -> List[int]:
    """Returns the line numbers from the function where calls to action_list functions occur."""
    lines = func_source.split('\n')
    action_lines = []
    for idx, line in enumerate(lines):
        # Check if any function in the action_list is called in the line
        if any(re.search(r'\b{}\b'.format(action), line) for action in action_list):
            action_lines.append(idx)  # Store the index of the action line
    return action_lines

# Generate the intermediate steps for each hash ID
intermediate_dataset = []

for hash_id, task in tasks_data.items():
    try:
        # Get a random train input example (now pick the first one for consistency)
        train_examples = task['train']
        random_example = train_examples[0]  # Always pick the first example
        input_grid = tuple(tuple(row) for row in random_example['input'])

        # Get the corresponding transform function source code
        func_source = get_transform_function_source(hash_id)

        # Parse the function for lines that call grid functions from action_list
        action_line_indices = parse_action_calls(func_source, action_list)

        # If there are no intermediate grid function calls, treat the last line as an action point
        if not action_line_indices:
            action_line_indices = [len(func_source.split('\n')) - 2]  # Last non-empty line

        # Extract the function up to each action line and generate intermediate grids
        intermediate_steps = []
        for idx, line_idx in enumerate(action_line_indices):
            # Create a truncated version of the function by stopping at the action line
            truncated_func_lines = func_source.split('\n')[:line_idx + 1]
            action_line = truncated_func_lines[-1]
            truncated_func_lines.append(f"    return {action_line.split('=')[0].strip()}")  # Return the variable from the action line
            truncated_func_source = '\n'.join(truncated_func_lines).replace(f'verify_{hash_id}', 'transform_grid')

            # Debug: Show the truncated function being executed
            print(f"Executing truncated function for {hash_id} at step {idx + 1}, line {line_idx + 1}")
            print(truncated_func_source)

            # Execute the truncated function and get the intermediate grid
            local_vars = {}
            exec(truncated_func_source, globals(), local_vars)
            transform_grid = local_vars['transform_grid']
            intermediate_result = transform_grid(input_grid)

            # Store the intermediate step
            intermediate_steps.append({
                "step": idx + 1,
                "line_number": line_idx + 1,
                "line": action_line.strip(),
                "grid": compress_grid(intermediate_result),  # Compress grid
                "truncated_function": truncated_func_source
            })

        # Capture the final output grid (the complete function execution)
        final_func = func_source.replace(f'verify_{hash_id}', 'transform_grid')
        local_vars = {}
        exec(final_func, globals(), local_vars)
        transform_grid = local_vars['transform_grid']
        final_output_grid = transform_grid(input_grid)

        # Create an entry for this task with intermediate steps
        intermediate_entry = {
            "hash_id": hash_id,
            "input_grid": compress_grid(input_grid),  # Compress input grid
            "transform_function": func_source,
            "intermediate_steps": intermediate_steps,
            "final_output_grid": compress_grid(final_output_grid),  # Compress output grid
            "expected_output_grid": compress_grid(random_example['output'])
        }

    except Exception as e:
        # In case of error, store minimal entry
        print(f"Error processing hash ID {hash_id}: {e}")
        try:
            final_func = func_source.replace(f'verify_{hash_id}', 'transform_grid')
            local_vars = {}
            exec(final_func, globals(), local_vars)
            transform_grid = local_vars['transform_grid']
            final_output_grid = transform_grid(input_grid)

            intermediate_entry = {
                "hash_id": hash_id,
                "input_grid": compress_grid(input_grid),
                "final_output_grid": compress_grid(final_output_grid)
            }
        except Exception as final_error:
            intermediate_entry = {
                "hash_id": hash_id,
                "input_grid": compress_grid(input_grid),
                "error": str(final_error)
            }

    intermediate_dataset.append(intermediate_entry)

# Save the intermediate dataset to a new JSON file
output_file_path = 'data/intermediate_grids_dataset.json'
with open(output_file_path, 'w') as f:
    json.dump(intermediate_dataset, f, indent=4)

print(f"Intermediate dataset generated and saved to {output_file_path}")

