import re

def inline_statements(function_code):
    """
    Converts a function with intermediate variables into a one-liner return statement.
    """
    # Extract all variable assignments (x0 = ..., x1 = ..., etc.)
    assignments = re.findall(r'(x\d+\s*=\s*.+)', function_code)
    
    # Extract the return statement (return xN)
    return_match = re.search(r'return\s+(x\d+)', function_code)
    
    if not return_match:
        print("No return statement found in the function.")
        return None
    
    return_var = return_match.group(1)
    
    # Create a dictionary to map variable names to their expressions
    var_map = {}
    for assign in assignments:
        var, expr = assign.split('=', 1)
        var = var.strip()
        expr = expr.strip()
        var_map[var] = expr
    
    # Debug: Print variable mapping
    # print(f"Variable Mapping: {var_map}")
    
    # Recursive function to inline variables
    def inline(var):
        if var not in var_map:
            return var
        expr = var_map[var]
        # Find all variables within the expression
        nested_vars = re.findall(r'\bx\d+\b', expr)
        for nv in nested_vars:
            # Replace the variable with its inlined expression, recursively
            expr = expr.replace(nv, f"({inline(nv)})")
        return expr
    
    # Get the fully inlined return expression
    inlined_return = inline(return_var)
    
    # Construct the new return statement
    return_statement = f"return {inlined_return}"
    return return_statement

def process_functions_block(functions_content):
    """
    Processes a block of functions and converts each into a one-liner.
    """
    # Split the functions based on the 'def verify_' pattern using a lookahead
    function_splits = re.split(r'(?=def\s+verify_[\da-f]+\s*\()', functions_content)
    
    converted_functions = []
    
    for func in function_splits:
        func = func.strip()
        if not func:
            continue  # Skip empty strings
        
        # Ensure the function starts with 'def verify_'
        if not func.startswith('def verify_'):
            print(f"Skipping unexpected content:\n{func[:50]}...")
            continue
        
        # Separate the header and the body
        header_match = re.match(r'(def\s+verify_[\da-f]+\s*\(.*?\)\s*->\s*Grid\s*:\s*)', func)
        if not header_match:
            print(f"Could not parse function header for:\n{func[:50]}...")
            continue
        
        header = header_match.group(1)
        body = func[len(header):].strip()
        
        # Inline the function
        inlined_return = inline_statements(body)
        if not inlined_return:
            print(f"Failed to inline function:\n{header}")
            continue
        
        # Construct the new function with the inlined return
        new_function = f"{header}    {inlined_return}"
        converted_functions.append(new_function)
    
    return converted_functions

def process_verifiers_file(input_file, output_file):
    """
    Reads the input file, processes all functions, and writes the output to a new file.
    """
    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        return
    
    # Define markers
    start_marker = "####startfunctions####"
    end_marker = "####endfunctions####"
    
    # Extract content between the markers
    functions_block_match = re.search(
        f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}", 
        content, 
        re.DOTALL
    )
    
    if not functions_block_match:
        print("No functions block found between the specified markers.")
        return
    
    functions_content = functions_block_match.group(1).strip()
    
    # Process the functions block
    converted_functions = process_functions_block(functions_content)
    
    if not converted_functions:
        print("No functions were converted.")
        return
    
    # Reconstruct the final output with markers and two-line spacing between functions
    final_output = f"{start_marker}\n\n"
    final_output += "\n\n".join(converted_functions)
    final_output += f"\n\n{end_marker}"
    
    # Write the output to the new file
    with open(output_file, 'w') as f:
        # Write any content before the start_marker
        pre_marker = content.split(start_marker)[0]
        f.write(pre_marker.strip() + "\n\n")
        # Write the converted functions with markers
        f.write(final_output)
        # Optionally, append any content after the end_marker
        post_marker = content.split(end_marker)[-1]
        if post_marker.strip():
            f.write("\n\n" + post_marker.strip())
    
    print(f"Conversion completed. Output written to {output_file}")

if __name__ == "__main__":
    # Define input and output file paths
    input_file = 'functions.py'          # Replace with your actual input file path
    output_file = 'functions_one_liners.py'  # Replace with your desired output file path
    
    process_verifiers_file(input_file, output_file)

