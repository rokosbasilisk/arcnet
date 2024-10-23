import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from dsl import *  # Import the DSL functions you've defined

# Define grid dimensions and constants
GRID_SIZE = (30, 30)
BLACK = 0

# Helper function to display a grid
def display_grid(grid: Grid, title: str):
    """Display a grid using matplotlib with lines separating cells."""
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='tab10', vmin=0, vmax=10)
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)  # Hide tick marks
    st.pyplot(fig)

# Create the Streamlit app interface
st.title("Grid Transformer Visualizer")

# Input section for user-defined transformation function
st.header("Define a Transformation Function")
code_input = st.text_area(
    "Enter your transformation function (Python syntax):",
    value="""def transform_grid(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    if not x0:
        return I  # Return the input grid if no objects are found
    x1 = totuple(x0)
    x2 = apply(normalize, x1)
    x3 = mostcommon(x2) if x2 else frozenset()  # Default to an empty set if x2 is empty
    x4 = mostcolor(I) if palette(I) else 0  # Use color 0 if the palette is empty
    x5 = shape(x3) if x3 else (1, 1)  # Default shape to (1, 1) if no object is found
    x6 = canvas(x4, x5)
    x7 = paint(x6, x3)
    return x7"""
)

# Button to execute the transformation
if st.button("Run Transformation"):
    try:
        # Create an empty 30x30 grid filled with BLACK (0)
        input_grid = canvas(BLACK, GRID_SIZE)
        
        # Display input grid
        st.subheader("Input Grid")
        display_grid(input_grid, "Input Grid")

        # Compile and execute the user-defined function
        exec(code_input, globals())

        # Apply the user's transform_grid function to the input grid
        output_grid = transform_grid(input_grid)

        # Display output grid
        st.subheader("Output Grid")
        display_grid(output_grid, "Output Grid")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Instructions for the user
st.write("""
### Instructions
1. Write a Python function named `transform_grid` that takes a grid as input and returns a transformed grid.
2. Use the DSL functions from `dsl.py` to manipulate the grid.
3. Click the 'Run Transformation' button to see the input and output grids.
4. The input grid is a 30x30 grid filled with the color 0 (black).
5. Make sure your function handles empty inputs and potential errors gracefully.
""")

