from copy import deepcopy
import numpy as np

def out_of_bounds(node, x_bound, y_bound):
    if node[0] >= x_bound or node[0] < 0:
        return True
    elif node[1] >= y_bound or node[1] < 0:
        return True
    return False

class Vector():
    def __init__(self, startpt, endpt):
        self.startpt = startpt
        self.endpt = endpt

def generate_steps(positionA, positionB, step_size):
    vector = positionB - positionA
    theta = np.arctan2(vector[1], vector[0])
    x_step = step_size * np.cos(theta)
    y_step = step_size * np.sin(theta)
    pts = [deepcopy(positionA)]
    while step_size < np.linalg.norm(pts[-1]-positionB):
        pts.append( np.array([ pts[-1][0]+x_step, pts[-1][1]+y_step ]) )
    if not np.allclose(pts[-1], positionB):
        pts.append(deepcopy(positionB))
    return pts

def determine_collision(pt, connectivity_grid):
    # Map pt to grid
    coord = pt.astype(int)
    try:
        if connectivity_grid[coord[0], coord[1]] == 0:
            # Yes collision
            return True
        else:
            # No collision
            return False
    except IndexError:
        # Out of Bounds. Count as collision
        return True

def determine_collisions(pts, connectivity_grid):
    for pt in pts:
        # Map pt to grid
        if determine_collision(pt, connectivity_grid):
            return True, pt
    return False, None

# Make it so ping only works with line of sight
def raycast(positionA, positionB, connectivity_grid, step_size):
    pts = generate_steps(positionA, positionB, step_size)
    # No collisions means we have line of sight
    collision, pt = determine_collisions(pts, connectivity_grid)
    if not collision:
        return True, None
    else:
        return False, pt

def line_of_sight(positionA, positionB, connectivity_grid, step_size):
    return raycast(positionA, positionB, connectivity_grid, step_size)[0]

# Check whether two paths are the same path
def check_path(pathA, pathB):
    for posA, posB in zip(pathA, pathB):
        # Removed positions
        if np.isnan(posA[0]) or np.isnan(posA[1]) or np.isnan(posB[0]) or np.isnan(posB[1]):
            # Check that all positions are removed
            if not(np.isnan(posA[0]) and np.isnan(posA[1]) and np.isnan(posB[0]) and np.isnan(posB[1])):
                return False
        else:
            if not np.allclose(posA, posB):
                return False
    return True