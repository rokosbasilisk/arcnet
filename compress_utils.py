def compress_grid(grid):
    """Compress a grid with dimensions and RLE."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    flattened = [str(cell) for row in grid for cell in row]
    compressed = []
    current_char = flattened[0]
    count = 1

    # Create the RLE string with count for each character.
    for char in flattened[1:]:
        if char == current_char:
            count += 1
        else:
            compressed.append(f"{current_char}x{count}")
            current_char = char
            count = 1
    compressed.append(f"{current_char}x{count}")

    # Prefix with dimensions.
    return f"{rows}x{cols}|" + ",".join(compressed)

def decompress_grid(compressed):
    """Decompress a grid from the optimized RLE format."""
    dims, rle_data = compressed.split('|')
    rows, cols = map(int, dims.split('x'))
    flat_list = []

    # Reconstruct the flattened list from RLE.
    for segment in rle_data.split(','):
        char, count = segment.split('x')
        flat_list.extend([int(char)] * int(count))

    # Convert the flattened list back into a 2D grid.
    return [flat_list[i * cols:(i + 1) * cols] for i in range(rows)]

