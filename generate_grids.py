# generate_grids.py

import json
import importlib.util
import dsl
from deterministic_utils import choice, unifint, randint, sample
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

input_file = 'generators_deterministic_expressions.json'
output_file = 'generated_grids.json'

# Load deterministic expressions from JSON
with open(input_file, 'r') as f:
    expressions = json.load(f)

def evaluate_expression(expr_str):
    """Evaluate the expression string with deterministic utilities and log steps."""
    allowed_functions = {
        'canvas': dsl.canvas,
        'fill': dsl.fill,
        'paint': dsl.paint,
        'dmirror': dsl.dmirror,
        'totuple': dsl.totuple,
        'choice': choice,
        'unifint': unifint,
        'randint': randint,
        'sample': sample,
        # Add more functions as needed
    }
    try:
        result = eval(expr_str, {"__builtins__": {}}, allowed_functions)
        logging.debug(f"Evaluated expression: {expr_str} -> Result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error evaluating expression '{expr_str}': {e}")
        return None

# Generate grids
grids = {}
for hash_id, expr_str in expressions.items():
    logging.info(f"Processing expression with ID {hash_id}")
    grid_data = evaluate_expression(expr_str)
    if grid_data:
        grids[hash_id] = grid_data

# Save generated grids to JSON
with open(output_file, 'w') as f:
    json.dump(grids, f, indent=4)

print(f"Generated grids have been saved to {output_file}")

