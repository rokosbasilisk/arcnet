import ast
import json
import re
from typing import Dict, Any, Tuple

# Define fixed difficulty bounds
FIXED_DIFF_LB = 0.2
FIXED_DIFF_UB = 0.8

def unifint_fixed(bounds: Tuple[int, int]) -> int:
    """
    Generates a uniformly random integer based on fixed difficulty bounds.
    """
    a, b = bounds
    d = (FIXED_DIFF_LB + FIXED_DIFF_UB) / 2  # Deterministic value
    return min(max(a, round(a + (b - a) * d)), b)

class SingleLineTransformer(ast.NodeTransformer):
    """
    Transforms generate_* functions into single-line expressions by eliminating intermediate variables.
    Handles functions returning dictionaries with 'input' and 'output' keys.
    Replaces unifint calls with fixed values.
    """
    def __init__(self):
        super().__init__()
        self.symbol_table = {}
        self.expr = None

    def visit_FunctionDef(self, node):
        if not node.name.startswith('generate_'):
            return node

        # Reset symbol_table and expr for each function
        self.symbol_table = {}
        self.expr = None

        # Process each statement in the function body
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                # Handle variable assignments
                target = stmt.targets[0]
                expr = self.visit(stmt.value)
                if isinstance(target, ast.Name):
                    var_name = target.id
                    self.symbol_table[var_name] = expr
                elif isinstance(target, ast.Tuple):
                    var_names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
                    if isinstance(stmt.value, ast.Tuple) and len(var_names) == len(stmt.value.elts):
                        for var, val in zip(var_names, stmt.value.elts):
                            self.symbol_table[var] = self.visit(val)
            elif isinstance(stmt, ast.For):
                # Handle for loops
                loop_expr = self.handle_for_loop(stmt)
                # Assign loop expression to 'go' if applicable
                self.symbol_table['go'] = loop_expr
            elif isinstance(stmt, ast.Return):
                # Handle return statements
                return_expr = self.visit(stmt.value)
                if isinstance(return_expr, ast.Dict):
                    # Replace 'gi' and 'go' with their expressions from symbol_table
                    for i, (key, value) in enumerate(zip(return_expr.keys, return_expr.values)):
                        if isinstance(key, ast.Constant):
                            if key.value == 'input' and 'gi' in self.symbol_table:
                                return_expr.values[i] = self.symbol_table['gi']
                            elif key.value == 'output' and 'go' in self.symbol_table:
                                return_expr.values[i] = self.symbol_table['go']
                self.expr = return_expr

        # After processing all statements, construct the final dict
        if self.expr:
            self.expr = self.expr
        else:
            # Fallback to empty dict if 'gi' or 'go' is missing
            self.expr = ast.Dict(keys=[], values=[])

        # Debugging: Print the symbol table and final expression
        print(f"Processing function: {node.name}")
        print("Symbol Table:")
        for var, expr in self.symbol_table.items():
            expr_str = ast.unparse(expr) if isinstance(expr, ast.AST) else str(expr)
            print(f"  {var}: {expr_str}")
        print("Final Expression:")
        print(ast.unparse(self.expr))
        print("-" * 40)

        return None  # Remove the original function body

    def handle_for_loop(self, node):
        """
        Transforms for loops into execute_loop(iterable, lambda args: body)
        """
        # Extract loop target
        if isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node.target, ast.Tuple):
            target = ", ".join([elt.id for elt in node.target.elts if isinstance(elt, ast.Name)])
        else:
            target = "unknown"

        # Extract iterable
        iter_expr = self.visit(node.iter)

        # Extract loop body expressions
        body_exprs = []
        for stmt in node.body:
            if isinstance(stmt, ast.If):
                # Handle if statements inside the loop
                condition = self.visit(stmt.test)
                # Assuming single statement inside if
                if len(stmt.body) != 1:
                    continue  # Skip complex bodies
                if_stmt = stmt.body[0]
                if isinstance(if_stmt, ast.Assign):
                    var_target = if_stmt.targets[0]
                    if isinstance(var_target, ast.Name):
                        var_name = var_target.id
                        value = self.visit(if_stmt.value)
                        # Create a call to if_then_else(condition, value, var)
                        if_then_else_call = ast.copy_location(
                            ast.Call(
                                func=ast.Name(id='if_then_else', ctx=ast.Load()),
                                args=[
                                    condition,  # condition
                                    value,      # value if True
                                    ast.Name(id=var_name, ctx=ast.Load())  # var if False
                                ],
                                keywords=[]
                            ),
                            stmt
                        )
                        body_exprs.append(if_then_else_call)
            elif isinstance(stmt, ast.Assign):
                # Handle assignments within the loop
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    var_name = target.id
                    value = self.visit(stmt.value)
                    # Create an assignment expression
                    assign_expr = ast.copy_location(
                        ast.Assign(
                            targets=[ast.Name(id=var_name, ctx=ast.Store())],
                            value=value
                        ),
                        stmt
                    )
                    body_exprs.append(assign_expr)
                elif isinstance(target, ast.Tuple):
                    var_names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
                    if isinstance(stmt.value, ast.Tuple) and len(var_names) == len(stmt.value.elts):
                        for var, val in zip(var_names, stmt.value.elts):
                            assign_expr = ast.copy_location(
                                ast.Assign(
                                    targets=[ast.Name(id=var, ctx=ast.Store())],
                                    value=self.visit(val)
                                ),
                                stmt
                            )
                            body_exprs.append(assign_expr)
            else:
                pass  # Handle other types if necessary

        # Create a lambda function for the loop body
        if isinstance(node.target, ast.Name):
            args = [ast.arg(arg=node.target.id, annotation=None)]
        elif isinstance(node.target, ast.Tuple):
            args = [ast.arg(arg=elt.id, annotation=None) for elt in node.target.elts if isinstance(elt, ast.Name)]
        else:
            args = [ast.arg(arg="unknown", annotation=None)]

        # If there are multiple body expressions, use ast.Tuple
        if len(body_exprs) > 1:
            lambda_body = ast.Tuple(elts=body_exprs, ctx=ast.Load())
        elif len(body_exprs) == 1:
            lambda_body = body_exprs[0]
        else:
            lambda_body = ast.Constant(value=None)

        lambda_func = ast.copy_location(
            ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=args,
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[]
                ),
                body=lambda_body
            ),
            node
        )

        # Create the execute_loop call
        execute_loop_call = ast.copy_location(
            ast.Call(
                func=ast.Name(id='execute_loop', ctx=ast.Load()),
                args=[
                    iter_expr,
                    lambda_func
                ],
                keywords=[]
            ),
            node
        )

        return execute_loop_call

    def visit_Call(self, node):
        """
        Replace unifint calls with fixed values.
        """
        self.generic_visit(node)  # Process arguments first
        if isinstance(node.func, ast.Name) and node.func.id == 'unifint':
            # Check if the first two arguments are diff_lb and diff_ub
            if (len(node.args) >= 3 and
                isinstance(node.args[0], ast.Name) and node.args[0].id == 'diff_lb' and
                isinstance(node.args[1], ast.Name) and node.args[1].id == 'diff_ub'):
                # Extract the bounds argument
                bounds_arg = node.args[2]
                if isinstance(bounds_arg, ast.Tuple) and len(bounds_arg.elts) == 2:
                    a_node, b_node = bounds_arg.elts
                    if isinstance(a_node, ast.Constant) and isinstance(b_node, ast.Constant):
                        a = a_node.value
                        b = b_node.value
                        # Compute the fixed unifint value
                        fixed_value = unifint_fixed((a, b))
                        # Replace the call with a Constant node
                        return ast.copy_location(
                            ast.Constant(value=fixed_value),
                            node
                        )
        return node

    def visit_Name(self, node):
        """
        Replace variable names with their expressions from symbol_table.
        """
        if node.id in self.symbol_table:
            return self.symbol_table[node.id]
        else:
            return node

    def visit_Attribute(self, node):
        """
        Reconstruct attribute accesses.
        """
        node.value = self.visit(node.value)
        return node

    def visit_Compare(self, node):
        """
        Reconstruct comparison operators.
        """
        node.left = self.visit(node.left)
        node.comparators = [self.visit(comp) for comp in node.comparators]
        node.ops = [self.get_comparison_operator_ast(op) for op in node.ops]
        return node

    def visit_BinOp(self, node):
        """
        Reconstruct binary operations.
        """
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        node.op = self.get_binary_operator_ast(node.op)
        return node

    def visit_UnaryOp(self, node):
        """
        Reconstruct unary operations.
        """
        node.operand = self.visit(node.operand)
        node.op = self.get_unary_operator_ast(node.op)
        return node

    def visit_Dict(self, node):
        """
        Handle dictionary literals.
        """
        return node  # Let ast.unparse handle it

    def visit_Constant(self, node):
        """
        Handle constant values.
        """
        return node

    def get_comparison_operator_ast(self, op):
        # Map comparison operator nodes to their corresponding AST operator instances
        comparison_operator_map = {
            ast.Lt: ast.Lt(),
            ast.LtE: ast.LtE(),
            ast.Gt: ast.Gt(),
            ast.GtE: ast.GtE(),
            ast.Eq: ast.Eq(),
            ast.NotEq: ast.NotEq(),
            ast.Is: ast.Is(),
            ast.IsNot: ast.IsNot(),
            ast.In: ast.In(),
            ast.NotIn: ast.NotIn(),
        }
        return comparison_operator_map.get(type(op), ast.Eq())  # Default to Eq if unknown

    def get_binary_operator_ast(self, op):
        # Map binary operator nodes to their corresponding AST operator instances
        binary_operator_map = {
            ast.Add: ast.Add(),
            ast.Sub: ast.Sub(),
            ast.Mult: ast.Mult(),
            ast.Div: ast.Div(),
            ast.Mod: ast.Mod(),
            ast.Pow: ast.Pow(),
            ast.FloorDiv: ast.FloorDiv(),
            ast.LShift: ast.LShift(),
            ast.RShift: ast.RShift(),
            ast.BitOr: ast.BitOr(),
            ast.BitXor: ast.BitXor(),
            ast.BitAnd: ast.BitAnd(),
            ast.MatMult: ast.MatMult(),
        }
        return binary_operator_map.get(type(op), ast.Add())  # Default to Add if unknown

    def get_unary_operator_ast(self, op):
        # Map unary operator nodes to their corresponding AST operator instances
        unary_operator_map = {
            ast.UAdd: ast.UAdd(),
            ast.USub: ast.USub(),
            ast.Not: ast.Not(),
            ast.Invert: ast.Invert(),
        }
        return unary_operator_map.get(type(op), ast.UAdd())  # Default to UAdd if unknown

def extract_hash_id(function_name: str) -> str:
    """
    Extracts the hash ID from the function name.
    """
    match = re.match(r'generate_([a-f0-9]+)', function_name)
    if match:
        return match.group(1)
    else:
        return function_name

def transform_generators(file_path: str) -> Dict[str, str]:
    """
    Transforms generate_* functions into single-line expressions and returns a dictionary.
    """
    with open(file_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    expressions = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('generate_'):
            # Instantiate a new transformer for each function
            transformer = SingleLineTransformer()
            transformer.visit(node)
            hash_id = extract_hash_id(node.name)
            # Convert the expression AST node to a string using ast.unparse
            expr_ast = transformer.expr
            expr_code = ast.unparse(expr_ast).strip() if expr_ast else "{}"
            expressions[hash_id] = expr_code

    return expressions

def main():
    input_file = 'generators.py'  # Path to your generators.py
    output_file = 'generators_expressions.json'

    expressions = transform_generators(input_file)

    with open(output_file, 'w') as f:
        json.dump(expressions, f, indent=4)

    print(f"Transformed expressions have been saved to {output_file}")

if __name__ == "__main__":
    main()

