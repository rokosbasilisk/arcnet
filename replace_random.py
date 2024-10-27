import ast
import json
from typing import Tuple

# Define fixed difficulty bounds for unifint
FIXED_DIFF_LB = 0.2
FIXED_DIFF_UB = 0.8

def unifint_fixed(bounds: Tuple[int, int]) -> int:
    a, b = bounds
    d = (FIXED_DIFF_LB + FIXED_DIFF_UB) / 2
    return min(max(a, round(a + (b - a) * d)), b)

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
    input_file = 'generators_raw_expressions.json'
    output_file = 'generators_deterministic_expressions.json'

    with open(input_file, 'r') as f:
        raw_expressions = json.load(f)

    deterministic_expressions = {}
    for hash_id, expr in raw_expressions.items():
        try:
            new_expr = replace_random_calls_ast(expr)
            deterministic_expressions[hash_id] = new_expr
        except Exception as e:
            print(f"Error transforming expression for {hash_id}: {e}")
            deterministic_expressions[hash_id] = expr  # Fallback to original expression if transformation fails

    with open(output_file, 'w') as f:
        json.dump(deterministic_expressions, f, indent=4)

    print(f"Deterministic expressions have been saved to {output_file}")

if __name__ == "__main__":
    main()

