import numpy as np

def wave_x(xs):
    return 0.1*np.sin(xs/10)

def wave_y(ys):
    return 0.1*np.sin(ys/5+20)
