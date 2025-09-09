import random
import json

def rotate_90(grid, direction: bool):
    if direction:
        return [list(reversed(col)) for col in zip(*grid)]
    else:
        return [list(col) for col in reversed(list(zip(*grid)))]

def mirror(grid, axis: bool):
    if axis:
        return [list(reversed(row)) for row in grid]
    else:
        return list(reversed(grid))

def make_square(grid, size, pos, color):
    # makes a square of given size in given pos in a grid with a specific color
    for value in range(size):
        for value2 in range(size):
            grid[pos[0] + value][pos[1] + value2] = color
    return grid

def fill_grid(grid, color):
    # Fills the entire grid with a specific color
    rows = len(grid)
    cols = len(grid[0])
    for r in range(rows):
        for c in range(cols):
            grid[r][c] = color
    return grid


possible_dsls = [rotate_90, mirror]
possible_shapes = [make_square, fill_grid]
possible_colors = list(range(1, 10))  # ARC colors typically 0-9, 0 often background

def gen_problem():
    func = random.choice(possible_dsls)
    grid_size = random.randint(3, 6)
    shape = random.choice(possible_shapes)
    shape_size = random.randint(1, 3)
    shape_color = random.choice(possible_colors)

    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    if shape == make_square:
        shape_pos = (random.randint(0, grid_size - shape_size), random.randint(0, grid_size - shape_size))
        grid = shape(grid, shape_size, shape_pos, shape_color)
    elif shape == fill_grid:
        grid = shape(grid, shape_color)

    new_grid = func(grid, random.choice([True, False]))

    task = {
        "train": [
            {
                "input": grid,
                "output": new_grid
            }
        ],
        "test": [
            {
                "input": grid,
                "output": new_grid # For now, output is the same as train for simplicity
            }
        ]
    }
    return task


if __name__ == "__main__":
    arc_task = gen_problem()
    print(json.dumps(arc_task, indent=4))