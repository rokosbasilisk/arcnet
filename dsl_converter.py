import ast
import astor
import json


class DSLSimplifier(ast.NodeTransformer):
    """
    A class to simplify the given AST (Abstract Syntax Tree) of a verifier function
    into a more linear representation by inlining simple expressions and reducing
    unnecessary intermediate variable assignments.
    """

    def visit_Assign(self, node):
        # Attempt to simplify assignments by inlining simple values where possible
        if isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            if isinstance(node.value, (ast.Call, ast.BinOp, ast.UnaryOp)):
                self.simplify_assignments[target_name] = node.value
        return node

    def visit_Name(self, node):
        # Replace variable names with inlined expressions if they have been simplified
        return self.simplify_assignments.get(node.id, node)

    def visit_FunctionDef(self, node):
        # Reset simplified assignments for each function to ensure fresh inlining
        self.simplify_assignments = {}
        node.body = [self.visit(stmt) for stmt in node.body]
        node.body = [stmt for stmt in node.body if not self.is_trivial_assignment(stmt)]
        return node

    def is_trivial_assignment(self, node):
        # Remove trivial assignments that are no longer used
        return isinstance(node, ast.Assign) and isinstance(node.value, (ast.Name, ast.Constant))


class DSLExpander(ast.NodeTransformer):
    """
    A class to expand the simplified AST back into a more verbose form.
    It reintroduces intermediate variables and steps similar to the original form.
    """
    def __init__(self):
        self.var_counter = 0

    def _generate_var(self):
        """ Generate a unique variable name. """
        self.var_counter += 1
        return f"x{self.var_counter}"

    def visit_Call(self, node):
        # Expand function calls into separate variables where necessary.
        if len(node.args) <= 1:
            return node

        new_statements = []
        args = []
        for arg in node.args:
            if isinstance(arg, (ast.Call, ast.BinOp, ast.UnaryOp)):
                var_name = self._generate_var()
                new_statements.append(ast.Assign(targets=[ast.Name(id=var_name)], value=arg))
                args.append(ast.Name(id=var_name))
            else:
                args.append(arg)

        new_call = ast.Call(func=node.func, args=args, keywords=node.keywords)
        var_name = self._generate_var()
        new_statements.append(ast.Assign(targets=[ast.Name(id=var_name)], value=new_call))
        return ast.Tuple(elts=[*new_statements, ast.Name(id=var_name)], ctx=ast.Load())

    def visit_FunctionDef(self, node):
        self.var_counter = 0
        expanded_body = []

        for stmt in node.body:
            result = self.visit(stmt)
            if isinstance(result, ast.Tuple):
                expanded_body.extend(result.elts[:-1])
                expanded_body.append(ast.Return(value=result.elts[-1]))
            else:
                expanded_body.append(result)

        node.body = expanded_body
        return node


def simplify_verifier_functions(source_code: str) -> str:
    """
    Simplify the verifier functions defined in the provided source code.
    """
    tree = ast.parse(source_code)
    simplifier = DSLSimplifier()
    simplified_tree = simplifier.visit(tree)
    return astor.to_source(simplified_tree)


def expand_simplified_functions(source_code: str) -> str:
    """
    Expand the simplified verifier functions defined in the provided source code.
    """
    tree = ast.parse(source_code)
    expander = DSLExpander()
    expanded_tree = expander.visit(tree)
    return astor.to_source(expanded_tree)


def generate_json(source_code: str) -> str:
    """
    Generate a JSON where keys are the function identifiers and values are the simplified programs.
    """
    tree = ast.parse(source_code)
    result = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('verify_'):
            # Get the identifier (part after 'verify_')
            identifier = node.name.split('_')[1]
            # Get the simplified version of the function
            simplified_code = simplify_verifier_functions(astor.to_source(node))
            result[identifier] = simplified_code

    return json.dumps(result, indent=4)


# Example usage
if __name__ == "__main__":
    # Read the contents of the verifiers.py
    with open("verifiers.py", "r") as file:
        source_code = file.read()

    # Generate JSON with simplified programs
    json_output = generate_json(source_code)
    with open("verifiers_simplified.json", "w") as json_file:
        json_file.write(json_output)
    print("Simplified JSON written to verifiers_simplified.json")

    # Simplify the entire source code
    simplified_code = simplify_verifier_functions(source_code)
    with open("verifiers_simplified.py", "w") as file:
        file.write(simplified_code)
    print("Simplified code written to verifiers_simplified.py")

    # Expand the simplified code back to the verbose form
    expanded_code = expand_simplified_functions(simplified_code)
    with open("verifiers_expanded.py", "w") as file:
        file.write(expanded_code)
    print("Expanded code written to verifiers_expanded.py")

