import ast
import json

DSL_FILE = 'dsl.py'
OUTPUT_FILE = 'data/dsl_context.json'

with open(DSL_FILE, 'r') as f:
    tree = ast.parse(f.read())

functions = []

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        func_name = node.name
        args = []
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = ast.unparse(arg.annotation) if arg.annotation else 'Any'
            args.append(f"{arg_name}: {arg_type}")
        return_type = ast.unparse(node.returns) if node.returns else 'Any'
        docstring = ast.get_docstring(node) or ""
        description = docstring.strip().split('\n')[0]  # One-line description

        functions.append({
            "name": func_name,
            "arguments": args,
            "return_type": return_type,
            "description": description
        })

with open(OUTPUT_FILE, 'w') as f:
    json.dump(functions, f, indent=4)

print(f"Extracted {len(functions)} functions to {OUTPUT_FILE}")
