import streamlit as st
import json
import ast
from PIL import Image
import io
import inspect
import importlib

def main():
    st.title("Grid Visualizer")

    st.write("Paste your transformed expression below and click 'Run' to visualize the grid.")
    expression = st.text_area("Expression", height=300)

    if st.button("Run"):
        try:
            # Dynamically import the 'dsl' module
            dsl = importlib.import_module('dsl')

            # Extract all callable functions from 'dsl' and add them to exec_namespace
            exec_namespace = {}
            for name, obj in inspect.getmembers(dsl, inspect.isfunction):
                exec_namespace[name] = obj

            # Add any additional necessary functions that are not part of 'dsl'
            # For example, if 'execute_loop' and 'if_then_else' are defined here
            # If they are part of 'dsl', they will already be included
            # If not, define them here
            if 'execute_loop' not in exec_namespace:
                def execute_loop(iterable, func):
                    """
                    Executes a loop over 'iterable', applying 'func' to each item.
                    'func' should accept the loop variables as arguments.
                    Returns the result of the last iteration.
                    """
                    result = None
                    for item in iterable:
                        if isinstance(item, tuple):
                            result = func(*item)
                        else:
                            result = func(item)
                    return result
                exec_namespace['execute_loop'] = execute_loop

            if 'if_then_else' not in exec_namespace:
                def if_then_else(condition, output1, output2):
                    return output1 if condition else output2
                exec_namespace['if_then_else'] = if_then_else

            # Execute the pasted expression
            # Assuming the expression evaluates to a result directly
            result = eval(expression, exec_namespace)

            if result is not None:
                # Display the grid; adjust based on the type of 'result'
                if isinstance(result, Image.Image):
                    # If result is a PIL Image
                    buf = io.BytesIO()
                    result.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.image(byte_im, caption='Generated Grid', use_column_width=True)
                elif isinstance(result, dict):
                    # If result is a dictionary, display its contents
                    st.json(result)
                elif isinstance(result, list):
                    # If result is a list, display it as a table
                    st.write(result)
                else:
                    # Fallback to writing the result
                    st.write(result)
            else:
                st.error("No recognizable result found in the expression.")
        except Exception as e:
            st.error(f"Error while executing the expression: {e}")

if __name__ == "__main__":
    main()

