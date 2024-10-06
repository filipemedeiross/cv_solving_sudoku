import numpy as np


def reframe(points):
    points = points.reshape((4, 2))
    reordered_points = np.zeros((4, 1, 2), dtype=np.float32)

    point_sums = points.sum(axis=1)
    point_diff = np.diff(points, axis=1)

    reordered_points[0] = points[np.argmin(point_sums)]
    reordered_points[3] = points[np.argmax(point_sums)]
    reordered_points[1] = points[np.argmin(point_diff)]
    reordered_points[2] = points[np.argmax(point_diff)]

    return reordered_points
