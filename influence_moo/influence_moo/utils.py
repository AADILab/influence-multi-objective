from copy import deepcopy
import numpy as np
from numba import jit

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
    dist=pts[-1]-positionB
    dist=dist[0]+dist[1]
    while step_size < dist:
        pts.append( np.array([ pts[-1][0]+x_step, pts[-1][1]+y_step ]) )
        dist=pts[-1]-positionB
        dist=dist[0]+dist[1]
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
def raycast_slow(positionA, positionB, connectivity_grid, step_size):
    pts = generate_steps(positionA, positionB, step_size)
    # No collisions means we have line of sight
    collision, pt = determine_collisions(pts, connectivity_grid)
    if not collision:
        return True, None
    else:
        return False, pt


def raycast(positionA, positionB, connectivity_grid, step_size):
    pt=raycast_helper(positionA[0],positionA[1],positionB[0],positionB[1],connectivity_grid)
    if (pt==np.array([-100.0,-100.0])).all():
        return True, None
    else:
        return False, pt

@jit(nopython=True)
def raycast_helper(ax, ay, bx, by, grid):
    step = 0.1
    dist_travelled = 0.0
    dx = bx - ax
    dy = by - ay
    pt = np.array([-100.0, -100.0], dtype=np.float64)  # Predefine NumPy array with explicit dtype

    r = np.sqrt(dx * dx + dy * dy)
    if r==0:
        return pt
    dx /= r
    dy /= r
    rayx = ax
    rayy = ay

    while dist_travelled < r:
        dist_travelled += step
        rayx += dx * step
        rayy += dy * step

        # Ensure array indexing and comparison are separated
        rayx_int = int(rayx)
        rayy_int = int(rayy)
        if rayx_int >= 0 and rayx_int < grid.shape[0] and rayy_int >= 0 and rayy_int < grid.shape[1]:
            if grid[rayx_int, rayy_int] == 1:
                pt[0] = rayx
                pt[1] = rayy
                return pt
    return pt


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