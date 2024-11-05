import os
import ast
import json
import argparse
from typing import Dict, Tuple

def extract_functions(file_path: str, prefix: str) -> Dict[str, str]:
    """
    Extracts functions from a Python file that start with a specific prefix.

    Args:
        file_path (str): Path to the Python file.
        prefix (str): Prefix of the function names to extract.

    Returns:
        Dict[str, str]: A dictionary mapping hash to function body.
    """
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Parse the file content into an AST
    tree = ast.parse(file_content)

    functions = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith(prefix):
            # Extract the hash from the function name
            hash_code = node.name[len(prefix):]
            # Extract the function body as source code
            # We'll use ast.get_source_segment to retrieve the exact source
            # Note: Requires Python 3.8+
            try:
                function_body = ast.get_source_segment(file_content, node)
                if function_body is None:
                    raise ValueError(f"Could not extract source for function {node.name}")
                # Remove the def line
                body_lines = function_body.split('\n')[1:]
                # Remove any return statements
                body_lines = [line for line in body_lines if not line.strip().startswith('return')]
                # Strip leading/trailing whitespace from each line
                body_clean = '\n'.join([line.strip() for line in body_lines if line.strip()])
                functions[hash_code] = body_clean
            except Exception as e:
                print(f"Error processing function {node.name}: {e}")
                continue

    return functions

def create_json(generators_path: str, verifiers_path: str, output_path: str):
    """
    Creates a JSON file mapping each hash to its generator and verifier function bodies.

    Args:
        generators_path (str): Path to generators.py.
        verifiers_path (str): Path to verifiers.py.
        output_path (str): Path to save the output JSON.
    """
    generators = extract_functions(generators_path, prefix='generate_')
    verifiers = extract_functions(verifiers_path, prefix='verify_')

    print(f"Found {len(generators)} generator functions and {len(verifiers)} verifier functions.")

    # Find common hashes present in both generators and verifiers
    common_hashes = set(generators.keys()).intersection(set(verifiers.keys()))
    print(f"Number of common hashes: {len(common_hashes)}")

    data = {}
    for hash_code in common_hashes:
        data[hash_code] = {
            'generator': generators[hash_code],
            'verifier': verifiers[hash_code]
        }

    with open(output_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print(f"JSON data saved to {output_path}. Total entries: {len(data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare JSON dataset from generators.py and verifiers.py")
    parser.add_argument('--generators', type=str, default='re_arc/generators.py', help='Path to generators.py')
    parser.add_argument('--verifiers', type=str, default='re_arc/verifiers.py', help='Path to verifiers.py')
    parser.add_argument('--output', type=str, default='data/function_pairs.json', help='Output JSON file path')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_json(args.generators, args.verifiers, args.output)

