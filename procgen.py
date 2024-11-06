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
import traceback
import re

# Ensure the OpenAI API key is set
# It's recommended to set it as an environment variable for security reasons
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define paths
RE_ARC_PATH = 're_arc/generators.py'
GRIDS_DIR = 'grids'
FUNC_DEFS_DIR = 'func_defs'
FINAL_FUNC_DEFS_DIR = 'final_func_defs'  # New directory for final functions

# Create necessary directories if they don't exist
os.makedirs(GRIDS_DIR, exist_ok=True)
os.makedirs(FUNC_DEFS_DIR, exist_ok=True)
os.makedirs(FINAL_FUNC_DEFS_DIR, exist_ok=True)  # Ensure final functions directory exists

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

def print_grid_in_terminal(gi):
    """
    Print the 'gi' grid in the terminal using termcolor for colored output.
    Each cell is represented by two spaces with the background color corresponding to its value.
    """
    for row in gi:
        line = ''
        for cell in row:
            # Get the background color; default to 'on_black' if not found
            on_color = TERM_COLOR_MAP.get(cell, 'on_black')
            
            # Add two spaces with the background color
            try:
                line += colored('  ', on_color=on_color)
            except KeyError:
                # Handle any invalid color codes gracefully
                line += colored('  ', on_color='on_black')
        print(line)


# OpenAI API pricing (as of April 2023)
# Update these values based on the latest OpenAI pricing
MODEL_NAME = "gpt-4o-mini"  # Replace with the specific model name if different
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
    - Add print statements after each line to output the intermediate variable values 
      (only 'h', 'w', and variables obtained by randint, uniform, sample, choice).
    - Ensure the function returns 'gi' directly without intermediate variables.
    
    Returns the modified code and the cost of the API call.
    """
    prompt = f"""
Given the following Python function, modify it to:
1. Set diff_lb = 0 and diff_ub = 1.
2. Remove all code related to calculating 'go' and only keep the code necessary for calculating 'gi'.
3. Add a print statement after each line to output the values of variables 'h', 'w', and any variables obtained from randint, uniform, sample, or choice functions.
4. Eliminate all intermediate variables and return 'gi' directly as a single expression without wrapping it in a dictionary.

Here is the original function:

{function_code}

Provide the modified function code only without any code block markers.
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

        # Remove any potential code block markers
        if modified_code.startswith("```"):
            modified_code = '\n'.join(modified_code.split('\n')[1:])  # Remove first line
        if modified_code.endswith("```"):
            modified_code = '\n'.join(modified_code.split('\n')[:-1])  # Remove last line

        # Extract token usage
        usage = response['usage']
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        cost = (prompt_tokens * PROMPT_TOKEN_COST) + (completion_tokens * COMPLETION_TOKEN_COST)

        return modified_code, cost
    except Exception as e:
        print(f"Error modifying function code: {e}")
        traceback.print_exc()
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
            gi = modified_func()
        print_output = f.getvalue()
        
        return gi, print_output
    except Exception as e:
        print(f"Error executing modified function '{hash_id}': {e}")
        traceback.print_exc()
        return None, traceback.format_exc()

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
        traceback.print_exc()

def save_print_output(hash_id, print_output, modified_code):
    """
    Save the modified function code and the captured print output to a text file.
    """
    try:
        file_path = os.path.join(FUNC_DEFS_DIR, f"{hash_id}.txt")
        with open(file_path, 'w') as file:
            file.write("Modified Function Code:\n")
            file.write(modified_code)
            file.write("\n\nPrinted Outputs:\n")
            file.write(print_output)
    except Exception as e:
        print(f"Error saving print output for '{hash_id}': {e}")
        traceback.print_exc()

def inline_variables_in_function(original_modified_code, print_output):
    """
    Create a new function code by inlining the variables with their printed values.
    This function assumes that the print_output contains lines like 'var: value'.
    
    Returns the new function code as a string.
    """
    try:
        # Parse the printed outputs to extract variable values
        var_values = {}
        for line in print_output.strip().split('\n'):
            match = re.match(r'(\w+):\s+(.+)', line)
            if match:
                var, value = match.groups()
                # Attempt to convert value to int or float
                try:
                    if '.' in value:
                        var_values[var] = float(value)
                    else:
                        var_values[var] = int(value)
                except ValueError:
                    var_values[var] = value  # Keep as string if not a number

        # Debug: Print extracted variable values
        print(f"Extracted variable values: {var_values}")

        # Replace variables in the original_modified_code with their values
        # This assumes that variable assignments are simple and can be replaced
        # For more complex cases, a proper parser would be needed

        # Remove variable assignments and print statements
        lines = original_modified_code.split('\n')
        new_lines = []
        for line in lines:
            # Skip lines that assign to variables that have been inlined
            assign_match = re.match(r'\s*(\w+)\s*=\s*.*', line)
            if assign_match:
                var = assign_match.group(1)
                if var in var_values and var not in ['diff_lb', 'diff_ub']:
                    continue  # Remove this line as we'll inline its value
            # Skip print statements
            if 'print(' in line:
                continue
            new_lines.append(line)

        # Now, replace remaining variable usages with their values
        # This requires careful replacement to avoid unintended substitutions
        # We'll use regex word boundaries to replace whole words only
        modified_code = '\n'.join(new_lines)
        for var, value in var_values.items():
            if var in ['h', 'w', 'numcd', 'numc', 'fgc']:
                # Replace whole word occurrences
                # For strings, keep quotes
                if isinstance(value, str):
                    value_str = f'"{value}"'
                else:
                    value_str = str(value)
                modified_code = re.sub(rf'\b{var}\b', value_str, modified_code)

        # Now, ensure that the function returns gi directly
        # Since all variables are inlined, the expression should already be simplified
        return modified_code
    except Exception as e:
        print(f"Error inlining variables: {e}")
        traceback.print_exc()
        return original_modified_code  # Fallback to original if error occurs

def save_final_function(hash_id, final_code):
    """
    Save the final inlined function code to a separate directory.
    """
    try:
        file_path = os.path.join(FINAL_FUNC_DEFS_DIR, f"{hash_id}.py")
        with open(file_path, 'w') as file:
            file.write(final_code)
    except Exception as e:
        print(f"Error saving final function for '{hash_id}': {e}")
        traceback.print_exc()

def main():
    """
    Main function to process all generator functions.
    """
    global total_cost
    for hash_id, func in tqdm(function_dict.items(), desc="Processing functions"):
        # Get the original function code
        try:
            original_code = inspect.getsource(func)
            print(f"\nOriginal code for '{hash_id}':\n{original_code}\n")
        except Exception as e:
            print(f"Error retrieving source for function '{hash_id}': {e}")
            traceback.print_exc()
            continue
        
        # Modify the function code using OpenAI API
        modified_code, cost = modify_function_code(original_code)
        modified_code = "from re_arc.dsl import *\n" + modified_code   
        if modified_code:
            print(f"Modified code for '{hash_id}':\n{modified_code}\n")
        else:
            print(f"Skipping function '{hash_id}' due to modification error.\n")
            continue
        
        # Update total cost
        total_cost += cost
        
        # Execute the modified function and capture gi and print outputs
        gi, print_output = execute_modified_function(modified_code, hash_id)
        if gi is None:
            print(f"Skipping image creation for '{hash_id}' due to execution error.\n")
            continue
        
        # Print the grid in terminal using termcolor
        print(f"Grid for function '{hash_id}':")
        print_grid_in_terminal(gi)
        print()  # Add an empty line for better readability
        
        # Create and save the image from gi
        create_image_from_gi(gi, hash_id)
        
        # Save the modified function code and print outputs to a text file
        save_print_output(hash_id, print_output, modified_code)
        
        # Inline the variables in the function code
        final_code = inline_variables_in_function(modified_code, print_output)
        print(f"Final inlined code for '{hash_id}':\n{final_code}\n")
        
        # Save the final inlined function code to a separate file
        save_final_function(hash_id, final_code)
        
        # Print the cost for this function and the total cost so far
        print(f"Function '{hash_id}' processed. Cost for this function: ${cost:.6f}. Total cost so far: ${total_cost:.6f}.\n")

if __name__ == "__main__":
    main()

