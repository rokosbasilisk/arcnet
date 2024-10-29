import ast
import json
import re
import argparse
from typing import Dict, Tuple

# Define fixed difficulty bounds for unifint
FIXED_DIFF_LB = 0.2
FIXED_DIFF_UB = 0.8

def unifint_fixed(bounds: Tuple[int, int]) -> int:
    a, b = bounds
    d = (FIXED_DIFF_LB + FIXED_DIFF_UB) / 2
    return min(max(a, round(a + (b - a) * d)), b)

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
            expressions[hash_id] = expr_code

    return expressions

class ReplaceRandomCallsTransformer(ast.NodeTransformer):
    """AST transformer to replace random function calls with deterministic values."""

    def visit_Call(self, node):
        # Process innermost calls first
        self.generic_visit(node)

        # Replace `choice([...])` with the first element
        if isinstance(node.func, ast.Name) and node.func.id == 'choice':
            if isinstance(node.args[0], ast.List) and node.args[0].elts:
                return node.args[0].elts[0]  # Return first item of list

        # Replace `unifint(...)` with deterministic fixed value
        elif isinstance(node.func, ast.Name) and node.func.id == 'unifint':
            if len(node.args) == 3 and isinstance(node.args[2], ast.Tuple) and len(node.args[2].elts) == 2:
                a, b = node.args[2].elts
                if isinstance(a, ast.Constant) and isinstance(b, ast.Constant):
                    fixed_value = unifint_fixed((a.value, b.value))
                    return ast.Constant(value=fixed_value)

        # Replace `randint(a, b)` with the lower bound (a)
        elif isinstance(node.func, ast.Name) and node.func.id == 'randint':
            if len(node.args) == 2 and isinstance(node.args[0], ast.Constant):
                return node.args[0]

        # Replace `sample([...], k)` with first k elements of the list
        elif isinstance(node.func, ast.Name) and node.func.id == 'sample':
            if isinstance(node.args[0], ast.List) and isinstance(node.args[1], ast.Constant):
                k = node.args[1].value
                sample_elements = node.args[0].elts[:k]
                return ast.List(elts=sample_elements, ctx=ast.Load())

        return node  # Return modified or original node

def replace_random_calls_ast(expression: str) -> str:
    """Parse and replace random calls in the expression string."""
    try:
        tree = ast.parse(expression, mode='eval')  # Parse as an expression
        transformer = ReplaceRandomCallsTransformer()
        transformed_tree = transformer.visit(tree)
        ast.fix_missing_locations(transformed_tree)
        return ast.unparse(transformed_tree)  # Return modified expression as a string
    except SyntaxError as e:
        print(f"Error while processing expression: {expression}\nError: {e}")
        return expression

def main():
    parser = argparse.ArgumentParser(description="Transform generator functions into expressions.")
    parser.add_argument(
        '--replace_random',
        action='store_true',
        help='Replace random function calls with deterministic values.'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='generators.py',
        help='Path to the input Python file containing generator functions.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='generated_expressions.json',
        help='Path to the output JSON file.'
    )

    args = parser.parse_args()
    replace_random = args.replace_random
    input_file = args.input_file
    output_file = args.output_file

    # Step 1: Transform generator functions into expressions
    expressions = transform_generators(input_file)

    if replace_random:
        # Step 2: Replace random calls with deterministic values
        deterministic_expressions = {}
        for hash_id, expr in expressions.items():
            try:
                new_expr = replace_random_calls_ast(expr)
                deterministic_expressions[hash_id] = new_expr
            except Exception as e:
                print(f"Error transforming expression for {hash_id}: {e}")
                deterministic_expressions[hash_id] = expr  # Fallback to original expression if transformation fails
        expressions = deterministic_expressions

    # Save the final expressions to the output file
    with open(output_file, 'w') as f:
        json.dump(expressions, f, indent=4)

    print(f"Expressions have been saved to {output_file}")

if __name__ == "__main__":
    main()

