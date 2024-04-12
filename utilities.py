import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import skimage


def basic_threshold_img(image, threshold):
    return (image < threshold).astype(np.uint8)


def automatic_threshold_img(image):
    # Set threshold automatically based on histogram using Otsu's binarization
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image


def threshold_at_cumulative_value(image, diff_threshold):
    # Calculate histogram
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

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


def color_quantization(image, color_count):
    (h, w) = image.shape[:2]
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=color_count, n_init='auto')
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    return quant


def color_with_most_lines(image):
    colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)

    # plt.figure()
    # plt.subplot(len(colors) * 2 + 1, 1, 1)
    # plt.imshow(image)

    line_counts = []
    for i, color in enumerate(colors):
        color_mask = np.uint8(255 * np.all(image == color, axis=-1))
        # Classic straight-line Hough transform
        tested_angles = np.linspace(np.deg2rad(-15 + 90), np.deg2rad(15 + 90), 360, endpoint=False)
        h, theta, d = skimage.transform.hough_line(color_mask, theta=tested_angles)

        # plt.subplot(len(colors) * 2 + 1, 1, 2 * i + 1 + 1)
        # plt.imshow(color_mask, cmap="gray")
        # plt.subplot(len(colors) * 2 + 1, 1, 2 * i + 2 + 1)
        peaks = skimage.transform.hough_line_peaks(h, theta, d)
        line_counts.append(len(peaks[0]))

        # for _, angle, dist in zip(*peaks):
        #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #     plt.axline((x0, y0), slope=np.tan(angle + np.pi / 2))

    i = np.argmin(line_counts)
    # print(line_counts)
    # plt.subplot(len(colors) * 2 + 1, 1, 2 * i + 1 + 1)
    # plt.scatter(5, 5, c=[[1, 0, 0]])
    # plt.show()

    color = colors[i]
    color_mask = np.uint8(255 * np.all(image == color, axis=-1))
    return color_mask


def calculate_accuracy(dataset, function):
    correct = 0
    total = 0
    for i, sample in enumerate(dataset):
        img, mask, n_stitches = sample
        n_stitches_pred = function(img)
        if n_stitches == n_stitches_pred:
            correct += 1
        total += 1
    return correct / total
