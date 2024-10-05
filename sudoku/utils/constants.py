import numpy as np


STEP = 3
N    = 9
V    = np.arange(1, N + 1)

SQR = []
for idx in range(N):
    p = idx // STEP * STEP
    SQR.append((p, p + STEP))
