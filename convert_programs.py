import re

# Function to substitute variables into a return statement
def inline_statements(function_code):
    # Extract variable assignments and the return variable
    assignments = re.findall(r'(x\d+\s*=\s*.+)', function_code)
    return_var = re.search(r'return\s+(x\d+)', function_code).group(1)
    
    # Create a dictionary to store assignments
    var_map = {}
    for assign in assignments:
        var, expr = assign.split('=', 1)
        var_map[var.strip()] = expr.strip()
    
    # Inline the variables recursively
    def inline(var):
        if var not in var_map:
            return var
        expr = var_map[var]
        # Replace any variables in the expression with their inlined forms
        for v in re.findall(r'\bx\d+\b', expr):
            expr = expr.replace(v, f"({inline(v)})")
        return expr

    # Inline the return statement
    return f"return {inline(return_var)}"

# Function to process the entire file
def process_verifiers_file(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Find all function definitions and their contents
    function_blocks = re.findall(r'(def\s+verify_[\da-f]+\(.*?\):\n(.*?)(?=def|$))', content, re.DOTALL)

    # Process each function to convert to a one-liner
    converted_functions = []
    for header, body in function_blocks:
        inlined_func = inline_statements(body)
        converted_functions.append(f"{header}\n    {inlined_func}\n")

    # Write the converted functions to the output file
    with open(output_file, 'w') as f:
        f.write("from dsl import *\n\n")
        f.writelines(converted_functions)

# Example usage
input_file = 'verifiers.py'
output_file = 'verifiers_one_liners.py'
process_verifiers_file(input_file, output_file)

