import os
import sys
import inspect
import importlib.util
import openai
from tqdm import tqdm
import io
import contextlib
from PIL import Image
from termcolor import colored

# Ensure the OpenAI API key is set
# It's recommended to set it as an environment variable for security reasons
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define paths
RE_ARC_PATH = 're_arc/generators.py'
GRIDS_DIR = 'grids'
FUNC_DEFS_DIR = 'func_defs'

# Create necessary directories if they don't exist
os.makedirs(GRIDS_DIR, exist_ok=True)
os.makedirs(FUNC_DEFS_DIR, exist_ok=True)

# Add re_arc directory to sys.path to import generators module
re_arc_dir = os.path.dirname(RE_ARC_PATH)
if re_arc_dir not in sys.path:
    sys.path.append(re_arc_dir)

# Import the generators module
spec = importlib.util.spec_from_file_location("generators", RE_ARC_PATH)
generators = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generators)

# Extract all functions from the generators module
functions = inspect.getmembers(generators, inspect.isfunction)

# Create a dictionary with hash IDs as keys and function objects as values
function_dict = {}
for func_name, func_obj in functions:
    if func_name.startswith("generate_"):
        hash_id = func_name.split("_")[1]
        function_dict[hash_id] = func_obj

# Define color mapping for numbers 0-9 for image creation
COLOR_MAP = {
    0: (255, 255, 255),   # White
    1: (0, 0, 0),         # Black
    2: (255, 0, 0),       # Red
    3: (0, 255, 0),       # Green
    4: (0, 0, 255),       # Blue
    5: (255, 255, 0),     # Yellow
    6: (255, 165, 0),     # Orange
    7: (128, 0, 128),     # Purple
    8: (0, 255, 255),     # Cyan
    9: (255, 192, 203)    # Pink
}

# Define color mapping for termcolor (background colors)
TERM_COLOR_MAP = {
    0: 'on_white',
    1: 'on_grey',
    2: 'on_red',
    3: 'on_green',
    4: 'on_blue',
    5: 'on_yellow',
    6: 'on_magenta',
    7: 'on_cyan',
    8: 'on_white',
    9: 'on_white'
}

# OpenAI API pricing (as of April 2023)
# Update these values based on the latest OpenAI pricing
MODEL_NAME = "gpt-4"  # Replace with the specific model name if different
# Example pricing for GPT-4-8k:
PROMPT_TOKEN_COST = 0.03 / 1000  # $0.03 per 1k prompt tokens
COMPLETION_TOKEN_COST = 0.06 / 1000  # $0.06 per 1k completion tokens

# Initialize total cost
total_cost = 0.0

def modify_function_code(function_code):
    """
    Modify the function code to:
    - Set diff_lb = 0 and diff_ub = 1.
    - Retain only the code necessary for calculating 'gi'.
    - Add print statements after each line to output the intermediate variable values.
    - Ensure the function returns 'gi' at the end.
    - Include necessary imports from 're_arc.dsl' and 're_arc.utils'.
    
    Returns the modified code and the cost of the API call.
    """
    prompt = f"""
Given the following Python function, modify it to:
1. Set diff_lb = 0 and diff_ub = 1.
2. Remove all code related to calculating 'go' and only keep the code necessary for calculating 'gi'.
3. Add a print statement after each line to output the intermediate variable values.
4. Ensure the function returns a dictionary with 'input': gi at the end.
5. Include the necessary imports from 're_arc.dsl' and 're_arc.utils' at the beginning of the function.

Here is the original function:

{function_code}

Provide the modified function code only.
"""

    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that modifies Python functions based on user instructions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0
        )
        modified_code = response.choices[0].message['content'].strip()

        # Extract token usage
        usage = response['usage']
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        cost = (prompt_tokens * PROMPT_TOKEN_COST) + (completion_tokens * COMPLETION_TOKEN_COST)

        return modified_code, cost
    except Exception as e:
        print(f"Error modifying function code: {e}")
        return None, 0.0

def execute_modified_function(modified_code, hash_id):
    """
    Execute the modified function code and capture the output of print statements.
    Returns the 'gi' grid and the captured print output.
    """
    # Prepare a namespace for execution
    exec_namespace = {}
    try:
        # Execute the modified code
        exec(modified_code, exec_namespace)
        
        # Assume the modified function has the same name
        func_name = [name for name in exec_namespace if name.startswith("generate_")][0]
        modified_func = exec_namespace[func_name]
        
        # Redirect stdout to capture print statements
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            result = modified_func()
        print_output = f.getvalue()
        
        # Extract 'gi' from the returned dictionary
        gi = result.get('input')
        return gi, print_output
    except Exception as e:
        print(f"Error executing modified function '{hash_id}': {e}")
        return None, str(e)

def create_image_from_gi(gi, hash_id):
    """
    Create a colored grid image from the 'gi' grid and save it as a PNG file.
    """
    try:
        height = len(gi)
        width = len(gi[0]) if height > 0 else 0
        image = Image.new('RGB', (width, height))
        pixels = image.load()
        
        for y in range(height):
            for x in range(width):
                color_number = gi[y][x]
                color = COLOR_MAP.get(color_number, (0, 0, 0))  # Default to black if undefined
                pixels[x, y] = color
        
        image_path = os.path.join(GRIDS_DIR, f"{hash_id}.png")
        image.save(image_path)
    except Exception as e:
        print(f"Error creating image for '{hash_id}': {e}")

def save_print_output(hash_id, print_output):
    """
    Save the captured print output to a text file.
    """
    try:
        file_path = os.path.join(FUNC_DEFS_DIR, f"{hash_id}.txt")
        with open(file_path, 'w') as file:
            file.write(print_output)
    except Exception as e:
        print(f"Error saving print output for '{hash_id}': {e}")

def print_grid_in_terminal(gi):
    """
    Print the 'gi' grid in the terminal using termcolor for colored output.
    Each cell is represented by two spaces with the background color corresponding to its value.
    """
    for row in gi:
        line = ''
        for cell in row:
            color = TERM_COLOR_MAP.get(cell, 'on_black')
            # Add two spaces with the background color
            line += colored('  ', color)
        print(line)

def main():
    """
    Main function to process all generator functions.
    """
    global total_cost
    for hash_id, func in tqdm(function_dict.items(), desc="Processing functions"):
        # Get the original function code
        try:
            original_code = inspect.getsource(func)
            print(f"original_code: {original_code}")
        except Exception as e:
            print(f"Error retrieving source for function '{hash_id}': {e}")
            continue
        
        # Modify the function code using OpenAI API
        modified_code, cost = modify_function_code(original_code)
        print(f"modified_code: {modified_code}")
        if not modified_code:
            print(f"Skipping function '{hash_id}' due to modification error.")
            continue
        
        # Update total cost
        total_cost += cost
        
        # Execute the modified function and capture gi and print outputs
        gi, print_output = execute_modified_function(modified_code, hash_id)
        if gi is None:
            print(f"Skipping image creation for '{hash_id}' due to execution error.")
            continue
        
        # Print the grid in terminal using termcolor
        print(f"Grid for function '{hash_id}':")
        print_grid_in_terminal(gi)
        
        # Create and save the image from gi
        create_image_from_gi(gi, hash_id)
        
        # Save the print outputs to a text file
        save_print_output(hash_id, print_output)
        
        # Print the cost for this function and the total cost so far
        print(f"Function '{hash_id}' processed. Cost for this function: ${cost:.6f}. Total cost so far: ${total_cost:.6f}.\n")

if __name__ == "__main__":
    main()

