import random
def rotate_90(grid,direction:bool):
    if direction:
        return [list(reversed(col)) for col in zip(*grid)]
    else:
        return [list(col) for col in reversed(list(zip(*grid)))]
def mirror(grid,axis:bool):
    if axis:
        return [list(reversed(row)) for row in grid]
    else:
        return list(reversed(grid))
def make_square(grid,size,pos):
    #makes a square of given size in given pos in a grid
    for value in range(size):
        for value2 in range(size):
            grid[pos[0]+value][pos[1]+value2]=1
    return grid


possible_dsls=[rotate_90,mirror]
possible_shapes=[make_square]
def gen_problem():
    func=random.choice(possible_dsls)
    grid_size=random.randint(3,6)
    shape=random.choice(possible_shapes)
    shape_size=random.randint(1,3)
    shape_pos=(random.randint(0,grid_size-shape_size),random.randint(0,grid_size-shape_size))
    grid=[[0 for _ in range(grid_size)]for _ in range(grid_size)]
    grid=shape(grid,shape_size,shape_pos)
    new_grid=func(grid,random.choice([True,False]))
    return grid,new_grid,func
grid,new_grid,func=gen_problem()
for row in grid:
    print(row)
for row in new_grid:
    print(row)