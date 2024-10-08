import numpy as np


SHAPE      = 450, 450
SHAPE_CELL =  32,  32
MIN_AREA   = (SHAPE[0] * SHAPE[1]) / 4

WARP_POINTS = np.float32([[  0,   0],
                          [450,   0],
                          [  0, 450],
                          [450, 450]])
