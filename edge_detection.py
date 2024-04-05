import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy


def detect_edges(img):
    edges = cv2.Canny(img, 50, 200)

    kernel = np.ones((2, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = scipy.ndimage.binary_fill_holes(edges)
    edges = skimage.morphology.thin(edges).astype(np.uint8)

    kernel = np.ones((1, 8), np.uint8)
    long_hor_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    edges = long_hor_edges - edges

    kernel = np.ones((2, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = skimage.morphology.thin(edges).astype(np.uint8)

    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_length = 70
    long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_contour_length]

    cv2.drawContours(img, long_contours, -1, (0, 100, 0), 2)
    # plt.imshow(img)
    # plt.show()

    return len(long_contours)


# create function that skeletonize, remove small objects, remove small holes, and thin the edges



