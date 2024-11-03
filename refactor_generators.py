import re

def process_generator_file(input_path, output_path):
    # Read the input file
    with open(input_path, 'r') as f:
        content = f.read()
    
    # Split into individual functions while keeping the full function definition
    functions = re.split(r'\n(?=def generate_)', content)
    
    # Process each function
    processed_functions = []
    # Keep the import statement
    processed_functions.append(functions[0].strip())
    
    for func in functions[1:]:  # Skip the first one as it's imports
        # Extract function name
        func_name = re.search(r'def (generate_[a-f0-9]+)', func).group(1)
        
        # Process function
        lines = func.split('\n')
        new_lines = []
        
        # Add new function definition
        new_lines.append(f'def {func_name}():')
        
        # Add diff_lb and diff_ub assignments
        new_lines.append('    diff_lb = 0')
        new_lines.append('    diff_ub = 1')
        
        # Add remaining lines until gi, skipping the original function definition
        for line in lines[1:]:  # Skip the original function definition line
            if line.strip():  # Skip empty lines
                new_lines.append(line)
                if 'gi = ' in line:
                    # Add return statement with same indentation
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + 'return gi')
                    break
        
        processed_functions.append('\n'.join(new_lines))
    
    # Write to output file with double newlines between functions
    with open(output_path, 'w') as f:
        f.write('\n\n\n'.join(processed_functions))

# Usage
input_file = 're_arc/generators.py'
output_file = 're_arc/generators_processed.py'
process_generator_file(input_file, output_file)
