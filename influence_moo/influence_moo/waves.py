import numpy as np
from influence_moo.utils import out_of_bounds, Vector

def sample_waves(connectivity_grid, x_granularity, y_granularity, wave_x, wave_y):
    x_bound = connectivity_grid.shape[0]
    y_bound = connectivity_grid.shape[1]

    xs = np.linspace(0, x_bound, x_granularity)
    ys = np.linspace(0, y_bound, y_granularity)

    vectors = []
    for x in xs:
        for y in ys:
            if not out_of_bounds([x,y], x_bound, y_bound) and connectivity_grid[int(x),int(y)] == 1:
                startpt = np.array([x,y])
                endpt = startpt + np.array([wave_x(x), wave_y(y)])
                vectors.append(Vector(startpt, endpt))

    return vectors
