import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy
from skimage.morphology import skeletonize, thin
from skimage.util import invert, img_as_float
from scipy import ndimage
from skimage.segmentation import active_contour
from skimage.filters import gaussian


def random_lul(mean: float, std: float, img):
    val = -1
    while val < 0:
        val = np.random.normal(mean, std)

    return np.round(val)


def detect_edges(img):
    # gray_img = color.rgb2gray(img) * 255
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # thr_img = threshold_at_cumulative_value(gray_img, 0.01)
    # thr_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

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


def detect_edges_2(img):
    edges = cv2.Canny(img, 50, 200)

    # kernel = np.ones((2, 3), np.uint8)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # edges = scipy.ndimage.binary_fill_holes(edges)
    # edges = skimage.morphology.thin(edges).astype(np.uint8)

    # kernel = np.ones((1, 8), np.uint8)
    # long_hor_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    # edges = long_hor_edges - edges

    # kernel = np.ones((2, 3), np.uint8)
    # edges = cv2.dilate(edges, kernel, iterations=1)
    # edges = skimage.morphology.thin(edges).astype(np.uint8)

    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_length = 70
    long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_contour_length]

    cv2.drawContours(img, long_contours, -1, (0, 100, 0), 2)
    # plt.imshow(img)
    # plt.show()

    return len(long_contours)


def create_feature_vector(gray_img):
    img_inv = invert(gray_img)
    skeleton = skeletonize(img_inv, method="lee")

    x = np.linspace(5, 424, 100)
    y = np.linspace(136, 50, 100)
    init = np.array([x, y]).T

    cntr = active_contour(gaussian(gray_img, 1), init, alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(gray_img, cmap=plt.cm.gray)
    ax.plot(init[:, 0], init[:, 1], "--r", lw=3)
    ax.plot(cntr[:, 0], cntr[:, 1], "-b", lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, gray_img.shape[1], gray_img.shape[0], 0])

    plt.show()

    # imlabel = skimage.measure.label(skeleton)
    # plt.imshow(imlabel, cmap='gray')
    # plt.show()
