import cv2 as cv
import numpy as np

"""
Params: -img: opencv image
"""
def get_lines_in_image(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img_gray, 30, 250)
    img_col = img.copy()
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 70, None, 70, 15)
    if lines is not None:
        for i, line in enumerate(lines):
            rho = line[0][0]
            theta = line[0][1]
            # stuff for showing the lines, for debugging
            # a = np.cos(theta)
            # b = np.sin(theta)
            # x0 = a * rho
            # y0 = b * rho
            # pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            # pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(img_col, (line[0, 0], line[0, 1]), (line[0, 2], line[0, 3]), (0, 255, 0), 3, cv.LINE_AA)
    # return img_col
    return img_col, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)