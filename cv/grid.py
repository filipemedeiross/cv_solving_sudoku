import cv2
import numpy as np
from .constants import SHAPE_CELL


def get_number(cell):
    contours, _ = cv2.findContours(cell,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return cell

    contour = max(contours, key=len)
    if len(contour) < 10:
        return cell

    x, y, w, h = cv2.boundingRect(contour)

    side_length = max(w, h)
    x_center = x + w // 2
    y_center = y + h // 2
    x_square = x_center - side_length // 2
    y_square = y_center - side_length // 2

    x_square = max(x_square, 0)
    y_square = max(y_square, 0)
    x_square_end = min(x_square + side_length, cell.shape[1])
    y_square_end = min(y_square + side_length, cell.shape[0])

    return cell[y_square : y_square_end,
                x_square : x_square_end]

def split_cells(image):
    return [cell
            for row  in np.vsplit(image, 9)
            for cell in np.hsplit(row  , 9)]

def get_grid(cells, model):
    grid = np.zeros(81, 'uint8')

    for idx, cell in enumerate(cells):
        if np.sum(cell) < 135000:
            continue

        cell = get_number(cell[5:-5, 5:-5])
        cell = cv2.resize(cell, SHAPE_CELL)
        cell = cell / 255.

        cell = np.expand_dims(cell, axis=[0, -1])
        pred = model.predict(cell, verbose=0).squeeze()

        label = np.argmax(pred)
        if pred[label] >= 0.5:
            grid[idx] = label

    return grid.reshape(9, 9)
