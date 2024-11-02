import logging
import json
import inspect
from re_arc import dsl
from re_arc.dsl import *
# -------------------------------
# Utility Functions
# -------------------------------

def populate_allowed_functions(module):
    """Dynamically add all functions and constants from a module to a dictionary."""
    allowed = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or isinstance(obj, (int, float, str, tuple, list)):
            allowed[name] = obj
    return allowed

def modified_unifint(*args):
    """Handle unifint with variable arguments."""
    if len(args) == 1:
        return unifint(args[0])  # Single argument, just pass it along
    elif len(args) == 3:
        lb, ub, bounds = args
        return unifint(bounds)  # Process with bounds if provided
    else:
        raise ValueError("unifint() received an unexpected number of arguments")

# -------------------------------
# Expression Evaluation
# -------------------------------

def evaluate_expression(expr_str):
    """Evaluate the expression string with deterministic utilities and log steps."""
    
    # Load functions and constants from dsl and deterministic_utils
    allowed_functions = populate_allowed_functions(dsl)
    allowed_functions.update({
        'choice': choice,
        'unifint': modified_unifint,  # Use modified unifint for variable args
        'randint': randint,
        'sample': sample,
        # Default placeholders for missing variables and constants
        'gi': dsl.canvas(0, (10, 10)),
        'go': dsl.canvas(0, (10, 10)),
        'linc': 1,
        'diff_lb': 0.2,    # Example lower bound for unifint
        'diff_ub': 0.8,    # Example upper bound for unifint
        'bgc': 0,          # Background color placeholder
        'barc': 1,         # Example color code for bar
        'bgc1': 0,         # Placeholder for additional color variables
        'bgc2': 1,
        # Built-ins needed for eval
        'len': len,
        'tuple': tuple,
        'set': set,
        'frozenset': frozenset,
    })

    try:
        result = eval(expr_str, {"__builtins__": {}}, allowed_functions)
        logging.debug(f"Evaluated expression: {expr_str} -> Result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error evaluating expression '{expr_str}': {e}")
        return None

# -------------------------------
# Main Grid Generation Function
# -------------------------------

def generate_grids(expressions_file, output_file):
    """
    Generate grids from expressions stored in expressions_file and save results to output_file.
    
    Args:
        expressions_file (str): Path to the JSON file with grid generation expressions.
        output_file (str): Path to the output JSON file for saving generated grids.
    """
    try:
        # Load expressions
        with open(expressions_file, 'r') as f:
            expressions = json.load(f)
        
        generated_grids = {}

        # Process each expression
        for expr_id, expr_str in expressions.items():
            logging.info(f"Processing expression with ID {expr_id}")
            
            # Evaluate the expression and store the generated grid
            generated_grid = evaluate_expression(expr_str)
            if generated_grid is not None:
                generated_grids[expr_id] = generated_grid
            else:
                logging.error(f"Failed to generate grid for expression ID {expr_id}")
        
        # Save generated grids to the output file
        with open(output_file, 'w') as f:
            json.dump(generated_grids, f, indent=4)

        logging.info(f"Generated grids have been saved to {output_file}")
    
    except Exception as e:
        logging.error(f"Failed to generate grids: {e}")

# -------------------------------
# Logging Configuration
# -------------------------------

logging.basicConfig(level=logging.INFO)

# -------------------------------
# Example Usage
# -------------------------------

generate_grids('data/generated_expressions.json', 'data/generated_grids.json')

