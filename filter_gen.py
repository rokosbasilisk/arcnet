import ast

def has_control_flow(node):
    """Recursively checks if the AST node contains control flow statements."""
    control_flow_nodes = (ast.If, ast.For, ast.While, ast.AsyncFor, ast.AsyncWith, ast.With)
    if isinstance(node, control_flow_nodes):
        return True
    for child in ast.iter_child_nodes(node):
        if has_control_flow(child):
            return True
    return False

def copy_functions_without_control_flow(input_file, output_file):
    with open(input_file, 'r') as f:
        source = f.read()
    tree = ast.parse(source)

    functions_to_copy = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if not any(has_control_flow(stmt) for stmt in node.body):
                functions_to_copy.append(node)

    # Unparse the functions and write them to the output file
    import astor  # You may need to install astor using pip install astor

    with open(output_file, 'w') as f:
        for func in functions_to_copy:
            func_code = astor.to_source(func)
            f.write(func_code + '\n')

if __name__ == '__main__':
    copy_functions_without_control_flow('generators.py', 'functions_without_control_flow.py')

