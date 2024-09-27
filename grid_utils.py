from typing import List, Tuple, Dict
from termcolor import colored

# Define type aliases for clarity
Grid = List[List[int]]
Action = Tuple[int, int, int]
ActionNumber = int

class GridDrawer:
    def __init__(self):
        """
        Initializes the GridDrawer with a default 30x30 grid filled with zeros.
        """
        self.max_size = 30
        self.default_value = 0
        self.padding_value = 10
        self.current_grid = self._create_default_grid()
        self.color_mapping = self._define_color_mapping()
        self.action_mapping: Dict[ActionNumber, Action] = {}
        self.next_action_number: ActionNumber = 1  # Start numbering actions from 1

    def _create_default_grid(self) -> Grid:
        """Creates a 30x30 grid filled with the default value (0)."""
        return [[self.default_value for _ in range(self.max_size)] for _ in range(self.max_size)]

    def _define_color_mapping(self) -> Dict[int, str]:
        """
        Defines a mapping from grid values to termcolor background colors.
        Only colors supported by termcolor are used.
        """
        # Safe color mapping using supported background colors
        return {
            0: 'on_blue',        # Substitute for 'on_black' or 'on_grey'
            1: 'on_green',
            2: 'on_yellow',
            3: 'on_magenta',
            4: 'on_cyan',
            5: 'on_red',
            6: 'on_green',
            7: 'on_yellow',
            8: 'on_magenta',
            9: 'on_cyan',
            10: 'on_white'        # Padding color
        }

    def pad_grid(self, grid: Grid) -> Grid:
        """
        Pads the given grid symmetrically with the padding value to make it 30x30.

        :param grid: The input grid to pad.
        :return: A 30x30 padded grid.
        """
        original_rows = len(grid)
        original_cols = len(grid[0]) if original_rows > 0 else 0

        if original_rows > self.max_size or original_cols > self.max_size:
            raise ValueError("Grid size exceeds maximum allowed size of 30x30.")

        # Initialize padded grid with padding_value
        padded_grid = [[self.padding_value for _ in range(self.max_size)] for _ in range(self.max_size)]

        # Calculate padding for rows and columns
        row_padding_top = (self.max_size - original_rows) // 2
        col_padding_left = (self.max_size - original_cols) // 2

        # Insert the original grid into the padded grid
        for i in range(original_rows):
            for j in range(original_cols):
                padded_grid[i + row_padding_top][j + col_padding_left] = grid[i][j]

        return padded_grid

    def encode_action(self, action: Action) -> ActionNumber:
        """
        Encodes an action tuple into a unique action number.

        :param action: The action tuple to encode.
        :return: The unique action number assigned to the action.
        """
        if action in self.action_mapping.values():
            # Find the existing action number
            for number, act in self.action_mapping.items():
                if act == action:
                    return number
        # Assign a new action number
        action_number = self.next_action_number
        self.action_mapping[action_number] = action
        self.next_action_number += 1
        return action_number

    def draw_state(self, initial_grid: Grid, final_grid: Grid) -> List[ActionNumber]:
        """
        Computes the list of actions to transform the initial grid to the final grid.
        Updates the current grid.

        :param initial_grid: The starting grid state.
        :param final_grid: The desired grid state after transformation.
        :return: A list of action numbers representing the actions.
        """
        padded_initial = self.pad_grid(initial_grid)
        padded_final = self.pad_grid(final_grid)

        actions = []
        for i in range(self.max_size):
            for j in range(self.max_size):
                initial_value = padded_initial[i][j]
                final_value = padded_final[i][j]
                if initial_value != final_value:
                    action = (i, j, final_value)
                    action_number = self.encode_action(action)
                    actions.append(action_number)
                    self.current_grid[i][j] = final_value

        return actions

    def print_grid(self, grid: Grid):
        """
        Prints the grid to the console with color representations using termcolor.

        :param grid: The grid to print.
        """
        if not isinstance(grid, list):
            print("Invalid grid: Not a list.")
            return

        for row_idx, row in enumerate(grid):
            if not isinstance(row, list):
                print(f"Invalid grid at row {row_idx}: Not a list.")
                return
            row_str = ''
            for cell in row:
                if not isinstance(cell, int):
                    print(f"Invalid cell value at row {row_idx}: {cell} is not an integer.")
                    return
                on_color = self.color_mapping.get(cell, 'on_white')  # Get background color
                try:
                    # Use on_color parameter correctly
                    row_str += colored('  ', on_color=on_color)  # Two spaces with background color
                except KeyError:
                    print(f"Warning: Background color '{on_color}' is not recognized. Using 'on_white' instead.")
                    row_str += colored('  ', on_color='on_white')  # Fallback to 'on_white'
            print(row_str)
        print("\n")  # Add extra newline for better readability

    def get_action_mapping(self) -> Dict[ActionNumber, Action]:
        """
        Returns the mapping of action numbers to action tuples.

        :return: A dictionary mapping action numbers to action tuples.
        """
        return self.action_mapping

# -------------------- Test Cases --------------------

def test_grid_drawer():
    """
    Tests the GridDrawer class using specific examples.
    Implements action encoding where each action number refers to one action tuple.
    """
    # Initialize GridDrawer
    drawer = GridDrawer()

    # Define the specific examples provided
    examples = [
        {
            'input': [
                [0, 7, 7],
                [7, 7, 7],
                [0, 7, 7]
            ],
            'output': [
                [0, 0, 0, 0, 7, 7, 0, 7, 7],
                [0, 0, 0, 7, 7, 7, 7, 7, 7],
                [0, 0, 0, 0, 7, 7, 0, 7, 7],
                [0, 7, 7, 0, 7, 7, 0, 7, 7],
                [7, 7, 7, 7, 7, 7, 7, 7, 7],
                [0, 7, 7, 0, 7, 7, 0, 7, 7],
                [0, 0, 0, 0, 7, 7, 0, 7, 7],
                [0, 0, 0, 7, 7, 7, 7, 7, 7],
                [0, 0, 0, 0, 7, 7, 0, 7, 7]
            ]
        },
        {
            'input': [
                [4, 0, 4],
                [0, 0, 0],
                [0, 4, 0]
            ],
            'output': [
                [4, 0, 4, 0, 0, 0, 4, 0, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 4, 0, 0, 0, 0, 0, 4, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 4, 0, 0, 0, 0]
            ]
        }
    ]

    for idx, example in enumerate(examples, start=1):
        print(f"=== Test Case {idx}: Using Provided Input Grid as Initial State ===\n")
        input_grid = example['input']
        final_grid = example['output']

        # Print Initial Grid
        print("Initial Grid:")
        drawer.print_grid(input_grid)

        # Print Final Grid
        print("Final Grid:")
        drawer.print_grid(final_grid)

        # Draw the actions
        try:
            actions = drawer.draw_state(input_grid, final_grid)
            print(f"Actions to transform initial to final state: {len(actions)} actions found.\n")
            print("List of Action Numbers:")
            print(actions)
            print("\nMapping of Action Numbers to Action Tuples:")
            for action_number in actions:
                action_tuple = drawer.get_action_mapping()[action_number]
                print(f"Action {action_number}: {action_tuple}")
        except Exception as e:
            print(f"Error processing Test Case {idx}: {e}\n")

        # Reset current grid for the next example
        drawer.current_grid = drawer.pad_grid(drawer._create_default_grid())

    # Test Case 3: Initial state is the default 30x30 grid filled with zeros
    print(f"=== Test Case {len(examples)+1}: Using Default 30x30 Grid as Initial State ===\n")
    default_initial_grid = drawer._create_default_grid()
    print("Initial Grid (Default 30x30 Grid):")
    drawer.print_grid(default_initial_grid)

    # Use the first example's output as the final grid for Test Case 3
    final_grid = examples[0]['output']
    print("Final Grid:")
    drawer.print_grid(final_grid)

    # Draw the actions
    try:
        actions = drawer.draw_state(default_initial_grid, final_grid)
        print(f"Actions to transform initial to final state: {len(actions)} actions found.\n")
        print("List of Action Numbers:")
        print(actions)
        print("\nMapping of Action Numbers to Action Tuples:")
        for action_number in actions:
            action_tuple = drawer.get_action_mapping()[action_number]
            print(f"Action {action_number}: {action_tuple}")
    except Exception as e:
        print(f"Error processing Test Case {len(examples)+1}: {e}\n")

    print("\nTesting completed.")

if __name__ == "__main__":
    test_grid_drawer()
