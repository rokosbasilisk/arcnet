import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import importlib
import inspect
from PIL import Image
import io
from typing import Tuple
from re_arc.dsl import *  # Import DSL functions
from re_arc.verifiers import *  # Import the verifiers containing transformation functions
from re_arc.deterministic_utils import *
# Define constants
GRID_SIZE = (30, 30)
BLACK = 0
DATA_FILE = 'data/arc-agi_training_challenges.json'

# Initialize state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

def main(mode="expression"):
    st.title("Multi-Purpose Grid App")

    if mode == "expression":
        run_expression_visualizer()
    elif mode == "transformer":
        run_grid_transformer()
    else:
        st.error("Invalid mode selected. Choose 'expression' or 'transformer'.")

def run_expression_visualizer():
    """Run the Expression Visualizer interface."""
    st.subheader("Expression Visualizer")

    st.write("Paste your transformed expression below and click 'Run' to visualize the grid.")
    expression = st.text_area("Expression", height=300)

    if st.button("Run"):
        try:
            dsl = importlib.import_module('re_arc.dsl')
            exec_namespace = {name: obj for name, obj in inspect.getmembers(dsl, inspect.isfunction)}

            if 'execute_loop' not in exec_namespace:
                exec_namespace['execute_loop'] = execute_loop
            if 'if_then_else' not in exec_namespace:
                exec_namespace['if_then_else'] = if_then_else

            result = eval(expression, exec_namespace)

            if result is not None:
                display_result(result)
            else:
                st.error("No recognizable result found in the expression.")
        except Exception as e:
            st.error(f"Error while executing the expression: {e}")

def run_grid_transformer():
    """Run the Grid Transformer Visualizer interface."""
    st.subheader("Grid Transformer Visualizer")

    # Load data
    data = load_data()
    hash_ids = list(data.keys())
    hash_id = get_current_hash_id(hash_ids)

    user_input_hash_id = st.text_input("Enter a hash ID to navigate to a specific task:", value=hash_id)
    if st.button("Go to Hash ID"):
        set_hash_id(hash_ids, user_input_hash_id)

    example_data = data[hash_id]['train'][0]['input']
    input_grid = tuple(tuple(row) for row in example_data)

    # Display input and output grids side by side using columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Grid")
        display_grid(input_grid, "Input Grid", fig_size=(3, 3))

    code_input = st.text_area(
        "Enter your transformation function (Python syntax):",
        value=load_transform_function_template(hash_id),
        height=300
    )

    if st.button("Run Transformation"):
        try:
            exec(code_input, globals())
            output_grid = transform_grid(input_grid)
            with col2:
                st.subheader("Output Grid")
                display_grid(output_grid, "Output Grid", fig_size=(3, 3))
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Navigation buttons
    navigation_controls(hash_ids)

    st.write("""
    ### Instructions
    1. Write or modify the Python function `transform_grid` to transform the grid.
    2. Use the DSL functions from `dsl.py`.
    3. Click 'Run Transformation' to view input and output grids.
    4. Use 'Previous' and 'Next' to navigate different hash IDs.
    """)

def load_data():
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def get_current_hash_id(hash_ids):
    return hash_ids[st.session_state.current_index]

def set_hash_id(hash_ids, hash_id):
    if hash_id in hash_ids:
        st.session_state.current_index = hash_ids.index(hash_id)
    else:
        st.error(f"Hash ID '{hash_id}' not found.")

def display_grid(grid: Tuple[Tuple[int]], title: str, fig_size=(4, 4)):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(grid, cmap='tab10', vmin=0, vmax=10)
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.set_title(title)
    st.pyplot(fig)

def load_transform_function_template(hash_id: str):
    func_name = f'verify_{hash_id}'
    if func_name in globals():
        func_source = inspect.getsource(globals()[func_name])
        return func_source.replace(func_name, 'transform_grid')
    return """def transform_grid(I: Grid) -> Grid:\n    return I"""

def display_result(result):
    if isinstance(result, Image.Image):
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.image(byte_im, caption='Generated Grid', use_column_width=True)
    elif isinstance(result, dict):
        st.json(result)
    elif isinstance(result, list):
        st.write(result)
    else:
        st.write(result)

def navigation_controls(hash_ids):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
    with col3:
        if st.button("Next") and st.session_state.current_index < len(hash_ids) - 1:
            st.session_state.current_index += 1

# Extra functions for expression execution
def execute_loop(iterable, func):
    result = None
    for item in iterable:
        result = func(*item) if isinstance(item, tuple) else func(item)
    return result

def if_then_else(condition, output1, output2):
    return output1 if condition else output2

# Run the main app with mode selection
if __name__ == "__main__":
    st.sidebar.title("App Mode")
    app_mode = st.sidebar.selectbox("Choose Mode", ["expression", "transformer"])
    main(mode=app_mode)

