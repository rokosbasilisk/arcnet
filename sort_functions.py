import ast
import os
from collections import defaultdict
from typing import Dict, List

def parse_functions(file_content: str) -> Dict[str, List[str]]:
    """
    Parses the given file content and extracts functions along with their return types.
    Returns a dictionary where keys are return types and values are lists of function definitions.
    """
    tree = ast.parse(file_content)
    functions = defaultdict(list)
    
    # Parse the AST to extract function definitions
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # Extract the function name and return type
            func_name = node.name
            return_type = None
            
            # Attempt to extract the return type annotation
            if node.returns:
                return_type = ast.unparse(node.returns)
            else:
                return_type = "Any"

            # Convert the function node back to its source code as a string
            func_code = ast.unparse(node)
            functions[return_type].append(func_code)
    
    return functions

def write_sorted_functions(functions: Dict[str, List[str]], output_file: str):
    """
    Writes the sorted functions into the output file with appropriate separators.
    """
    with open(output_file, 'w') as f:
        f.write("# types\nfrom constants import *\nfrom typing import Any, Tuple, Union\n\n")

        for return_type, funcs in sorted(functions.items()):
            f.write(f"# Functions returning {return_type}\n\n")
            for func in funcs:
                f.write(func + "\n\n")
            f.write(f"# End of functions returning {return_type}\n\n")

def main():
    input_file = "dsl.py"
    output_file = "transform_functions_sorted.py"
    
    # Read the input file content
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        return
    
    with open(input_file, 'r') as f:
        file_content = f.read()

    # Parse the functions and sort them by return type
    functions = parse_functions(file_content)
    
    # Write the sorted functions to the output file
    write_sorted_functions(functions, output_file)
    print(f"Sorted functions have been written to '{output_file}'.")

if __name__ == "__main__":
    main()

