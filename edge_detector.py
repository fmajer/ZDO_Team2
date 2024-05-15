import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy
from matplotlib import cm
from scipy.ndimage import median_filter
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)


def detect_edges(img):
    # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray_img = color.rgb2gray(img) * 255
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

    # cv2.drawContours(img, long_contours, -1, (0, 100, 0), 2)
    # plt.imshow(img)
    # plt.show()

    return len(long_contours)


def hough_vert_edge_detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img, 100, 200)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = scipy.ndimage.binary_fill_holes(edges).astype(np.uint8)

    kernel = np.ones((1, 4), np.uint8)
    long_hor_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges = edges - long_hor_edges

    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # lines = probabilistic_hough_line(edges, threshold=10, line_length=20, line_gap=7)

    tested_angles = np.linspace(np.deg2rad(-7), np.deg2rad(7), 360, endpoint=False)
    h, theta, d = skimage.transform.hough_line(edges, theta=tested_angles)

    plot_img_canny_hough(img, edges, lines=None, plot_hough=False)
    plt.show()

    plot_hough_lines(img, h, theta, d)
    plt.show()

    find_representative_lines(edges, h, theta, d)


def find_representative_lines(edges, h, theta, d):
    distance_threshold = 0.5  # Adjust this threshold as needed

    # Iterate over the lines and find out if they are close to each other
    close_lines = []
    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            distance = np.sqrt((h[i] - h[j]) ** 2 + (theta[i] - theta[j]) ** 2)
            if distance[0] < distance_threshold:
                close_lines.append((i, j))

    # dál to nemám a stejně to asi nebude fungovat


def plot_hough_lines(img, h, theta, d):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(img, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(img, cmap=cm.gray)
    origin = np.array((0, img.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax[2].plot(origin, (y0, y1), '-r')
    ax[2].set_xlim(origin)
    ax[2].set_ylim((img.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')


def plot_img_canny_hough(img, edges, lines, plot_hough):
    plt.figure(figsize=(15, 15))

    if plot_hough:
        plt.subplot(131)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title('Input image')

        plt.subplot(132)
        plt.imshow(edges,
                   cmap=plt.cm.gray
                   )
        plt.title('Canny edges')

        plt.subplot(133)
        plt.imshow(edges * 0)

        for line in lines:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]))

        plt.title('Hough')
        plt.axis('image')
        plt.show()
    else:
        plt.subplot(121)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title('Input image')

        plt.subplot(122)
        plt.imshow(edges,
                   cmap=plt.cm.gray
                   )
        plt.title('Canny edges')





