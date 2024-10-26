import inspect
import importlib.util
import os

# Load the module dynamically
module_path = "dsl.py"
module_name = os.path.splitext(os.path.basename(module_path))[0]

spec = importlib.util.spec_from_file_location(module_name, module_path)
dsl_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsl_module)

output_lines = []

# Iterate over each function in dsl.py
for name, func in inspect.getmembers(dsl_module, inspect.isfunction):
    # Extract the argument types
    signature = inspect.signature(func)
    params = [param.annotation for param in signature.parameters.values()]
    return_type = signature.return_annotation

    # Convert the parameter and return types to the required string format
    param_types = [p.__name__ if hasattr(p, "__name__") else "Numerical" for p in params]
    return_type_str = return_type.__name__ if hasattr(return_type, "__name__") else "Grid"

    # Construct the line for this function
    line = f'pset.addPrimitive({name}, [{", ".join(param_types)}], {return_type_str}, name="{name}")'
    output_lines.append(line)

# Write to a new script file
output_script_path = "generated_primitives.py"
with open(output_script_path, "w") as file:
    file.write("\n".join(output_lines))

print(f"Script has been generated and saved to {output_script_path}")

