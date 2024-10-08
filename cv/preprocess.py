import cv2
from .tools import *
from .constants import *


def preprocess(image):
    image_resized    = cv2.resize(image, SHAPE)
    image_gaussian   = cv2.GaussianBlur(image_resized, (3, 3), 6)
    image_threshold  = cv2.adaptiveThreshold(image_gaussian, 255, 1, 1, 11, 2)
    image_median     = cv2.medianBlur(image_threshold, 3)

    return image_median

def biggest_contour(image):
    contours, _ = cv2.findContours(image,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    contour  = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)

        if area < MIN_AREA:
            continue

        perimeter    = cv2.arcLength(c, True)
        approx_curve = cv2.approxPolyDP(c, 0.02* perimeter, True)

        if area > max_area and len(approx_curve) == 4:
            contour  = approx_curve
            max_area = area

    return contour

def get_perspective(image, shape, contour):
    p1 = reframe(contour)
    p2 = WARP_POINTS
    m = cv2.getPerspectiveTransform(p1, p2)

    return cv2.warpPerspective(image, m, shape)

def get_perspective_inv(image, shape, contour):
    p1 = reframe(contour)
    p2 = WARP_POINTS
    m = cv2.getPerspectiveTransform(p2, p1)

    return cv2.warpPerspective(image, m, shape)
