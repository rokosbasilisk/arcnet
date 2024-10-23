import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Tuple
from dsl import *  # Import DSL functions
from verifiers import *  # Import the verifiers containing transformation functions
import inspect

# Define constants
GRID_SIZE = (30, 30)
BLACK = 0

# Load the data from JSON
data = json.loads(open('data/arc-agi_training_challenges.json', 'r').read())
hash_ids = list(data.keys())

# State management for navigation
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

def get_current_hash_id():
    """Get the current hash ID based on the index."""
    return hash_ids[st.session_state.current_index]

def set_hash_id(hash_id: str):
    """Set the current index based on the hash ID provided."""
    if hash_id in hash_ids:
        st.session_state.current_index = hash_ids.index(hash_id)
    else:
        st.error(f"Hash ID '{hash_id}' not found. Please enter a valid hash ID.")

def display_grid(grid: Grid, title: str, fig_size=(4, 4)):
    """Display a grid using matplotlib with lines separating cells."""
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(grid, cmap='tab10', vmin=0, vmax=10)
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)  # Hide tick marks
    ax.set_title(title)
    st.pyplot(fig)

def load_transform_function_template(hash_id: str):
    """Dynamically load the transform function template based on hash ID."""
    func_name = f'verify_{hash_id}'
    if func_name in globals():
        # Extract the function's source code to display for editing
        func_source = inspect.getsource(globals()[func_name])
        # Replace the verifier function name with 'transform_grid'
        func_source = func_source.replace(func_name, 'transform_grid')
        return func_source
    else:
        # Provide a default template if no function is found
        return """def transform_grid(I: Grid) -> Grid:
    # Write your transformation logic here
    return I"""

# Create the Streamlit app interface
st.title("Grid Transformer Visualizer")
hash_id = get_current_hash_id()
st.subheader(f"Hash ID: {hash_id}")

# Input box for the user to type a hash ID and navigate directly to it
user_input_hash_id = st.text_input("Enter a hash ID to navigate to a specific task:", value=hash_id)

if st.button("Go to Hash ID"):
    set_hash_id(user_input_hash_id)

# Load the training data for the current hash ID
example_data = data[hash_id]['train'][0]['input']  # Example input grid

# Display the input grid in a smaller format
input_grid = tuple(tuple(row) for row in example_data)

# Display input and output grids side by side using columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("Input Grid")
    display_grid(input_grid, "Input Grid", fig_size=(3, 3))

# Load the default transformation logic into the text area
code_input = st.text_area(
    "Enter your transformation function (Python syntax):",
    value=load_transform_function_template(hash_id),
    height=300
)

# Button to execute the transformation
if st.button("Run Transformation"):
    try:
        # Compile and execute the user-defined function
        exec(code_input, globals())

        # Apply the user's transform_grid function to the input grid
        output_grid = transform_grid(input_grid)

        # Display the output grid in the second column
        with col2:
            st.subheader("Output Grid")
            display_grid(output_grid, "Output Grid", fig_size=(3, 3))

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Navigation buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("Previous") and st.session_state.current_index > 0:
        st.session_state.current_index -= 1

with col3:
    if st.button("Next") and st.session_state.current_index < len(hash_ids) - 1:
        st.session_state.current_index += 1

# Instructions for the user
st.write("""
### Instructions
1. The transformation function for each example is loaded from `verifiers.py` by default.
2. Write or modify the Python function `transform_grid` that takes a grid as input and returns a transformed grid.
3. Use the DSL functions from `dsl.py` to manipulate the grid.
4. Click the 'Run Transformation' button to see the input and output grids.
5. Use the 'Previous' and 'Next' buttons to navigate through different hash IDs and examples.
6. To navigate directly to a specific hash ID, enter the ID in the box and click 'Go to Hash ID'.
7. If an error occurs, it will be displayed below the grids.
""")

