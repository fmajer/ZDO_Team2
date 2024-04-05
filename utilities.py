import cv2
import numpy as np
import matplotlib.pyplot as plt


def basic_threshold_img(image, threshold):
    return (image < threshold).astype(np.uint8)


def automatic_threshold_img(image):
    # set threshold automatically based on histogram using Otsu's binarization
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image


def threshold_at_cumulative_value(image, diff_threshold):
    # Calculate histogram
    histogram = cv2.calcHist([image], [0], None, [256], [0,256])

    # Normalize the histogram
    histogram /= histogram.sum()

    # Calculate cumulative distribution function
    cdf = histogram.cumsum()

    # find where cumulative distribution function starts to rapidly rise and set threshold
    threshold = np.argmax(np.diff(cdf) > diff_threshold)

    # plot cdf and threshold in the same figure
    # plt.plot(cdf)
    # plt.axvline(x=threshold, color='r')
    # plt.show()

    return (image < threshold).astype(np.uint8)


def calculate_accuracy(dataset, function):
    correct = 0
    total = 0
    for i, sample in enumerate(dataset):
        img, gray_img, thr_img, mask, n_stitches = sample
        n_stitches_pred = function(img)
        if n_stitches == n_stitches_pred:
            correct += 1
        total += 1
    return correct / total
