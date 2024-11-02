import ast
import json
import re
import argparse
from typing import Dict

# Define fixed difficulty bounds for unifint
FIXED_DIFF_LB = 0
FIXED_DIFF_UB = 1

class SingleLineTransformer(ast.NodeTransformer):
    """
    Transforms generate_* functions into single-line expressions by eliminating intermediate variables.
    Preserves calls to random functions like choice, unifint, etc.
    """
    def __init__(self):
        super().__init__()
        self.symbol_table = {}
        self.expr = None

    def visit_FunctionDef(self, node):
        if not node.name.startswith('generate_'):
            return node

        self.symbol_table = {}
        self.expr = None

        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                target = stmt.targets[0]
                expr = self.visit(stmt.value)
                if isinstance(target, ast.Name):
                    self.symbol_table[target.id] = expr
            elif isinstance(stmt, ast.Return):
                return_expr = self.visit(stmt.value)
                if isinstance(return_expr, ast.Dict):
                    for i, (key, value) in enumerate(zip(return_expr.keys, return_expr.values)):
                        if isinstance(key, ast.Constant):
                            if key.value == 'input' and 'gi' in self.symbol_table:
                                return_expr.values[i] = self.symbol_table['gi']
                            elif key.value == 'output' and 'go' in self.symbol_table:
                                return_expr.values[i] = self.symbol_table['go']
                self.expr = return_expr

        if not self.expr:
            self.expr = ast.Dict(keys=[], values=[])

        return None

    def visit_Call(self, node):
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.symbol_table:
            return self.symbol_table[node.id]
        return node

def extract_hash_id(function_name: str) -> str:
    match = re.match(r'generate_([a-f0-9]+)', function_name)
    return match.group(1) if match else function_name

def transform_generators(file_path: str) -> Dict[str, str]:
    with open(file_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)
    expressions = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('generate_'):
            transformer = SingleLineTransformer()
            transformer.visit(node)
            hash_id = extract_hash_id(node.name)
            expr_code = ast.unparse(transformer.expr).strip() if transformer.expr else "{}"

            # Replace all instances of diff_lb and diff_ub with 0 and 1
            expr_code = expr_code.replace("diff_lb", "FIXED_DIFF_LB").replace("diff_ub", "FIXED_DIFF_UB")
            
            expressions[hash_id] = expr_code

    return expressions

def main():
    parser = argparse.ArgumentParser(description="Transform generator functions into expressions.")

    parser.add_argument(
        '--input_file',
        type=str,
        default='re_arc/generators.py',
        help='Path to the input Python file containing generator functions.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/generated_expressions.json',
        help='Path to the output JSON file.'
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    # Step 1: Transform generator functions into expressions
    expressions = transform_generators(input_file)

    # Step 2: Save the final expressions to the output file
    with open(output_file, 'w') as f:
        json.dump(expressions, f, indent=4)

    print(f"Expressions have been saved to {output_file}")

if __name__ == "__main__":
    main()

