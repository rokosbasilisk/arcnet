import ast
import json
import re
from typing import Dict

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

def main():
    input_file = 'generators.py'
    output_file = 'generators_raw_expressions.json'

    expressions = transform_generators(input_file)

    with open(output_file, 'w') as f:
        json.dump(expressions, f, indent=4)

    print(f"Raw expressions have been saved to {output_file}")

if __name__ == "__main__":
    main()

