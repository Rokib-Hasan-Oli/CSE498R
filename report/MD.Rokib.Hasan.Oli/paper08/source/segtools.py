"""Helpers shared by seg_check.py and charts.py: the authors' binarisation and contours (as in p08_plates.segment)."""
import cv2
import numpy as np


def binary_image(gray):
    """The first half of p08_plates.segment (the authors' code): 333x335, Otsu, erode/dilate, white frame."""
    lp = cv2.resize(gray, (333, 335))
    _, b = cv2.threshold(lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.dilate(cv2.erode(b, (3, 3)), (3, 3))
    b[0:3, :] = 255
    b[:, 0:3] = 255
    b[332:335, :] = 255
    b[:, 330:333] = 255
    return b


def blob(b, x, y, w, h):
    """A box of the binary image as the 24x44 white-bordered blob the authors feed to the CNN."""
    canvas = np.full((44, 24), 255, np.uint8)
    canvas[2:42, 2:22] = cv2.resize(b[y:y + h, x:x + w], (20, 40))
    return canvas


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1e-9, a[2] * a[3] + b[2] * b[3] - inter)


def all_contours(gray):
    """The 15 largest contours, unfiltered (the paper's order uses all of them)."""
    b = binary_image(gray)
    cntrs, _ = cv2.findContours(b.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]]
