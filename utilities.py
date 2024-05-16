import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import skimage
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


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

    initial_centers = np.linspace([0, 0, 0], [255, 255, 255], color_count, endpoint=True)
    # clt = MiniBatchKMeans(n_clusters=color_count, init=initial_centers, n_init="auto")
    clt = MiniBatchKMeans(n_clusters=color_count, n_init="auto")
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

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


def plot_images(images: list[any], rows: int, columns: int) -> None:
    plt.figure()

    for i, img in enumerate(images):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(img)
    pass


def train_classifiers(x_train_hog, y_train, x_val_hog, y_val):
    svm_classifier = SVC().fit(x_train_hog, y_train)
    svm_accuracy = svm_classifier.score(x_val_hog, y_val)

    gbc_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                         max_depth=1, random_state=0).fit(x_train_hog, y_train)
    gbc_accuracy = gbc_clf.score(x_val_hog, y_val)

    kn_classifier = KNeighborsClassifier(n_neighbors=5, algorithm='brute').fit(x_train_hog, y_train)
    kn_accuracy = kn_classifier.score(x_val_hog, y_val)

    ada_clf = AdaBoostClassifier(n_estimators=100, random_state=0).fit(x_train_hog, y_train)
    ada_accuracy = ada_clf.score(x_val_hog, y_val)

    mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000).fit(x_train_hog, y_train)
    mlp_accuracy = mlp_clf.score(x_val_hog, y_val)

    classifiers = [(svm_classifier, svm_accuracy), (gbc_clf, gbc_accuracy), (kn_classifier, kn_accuracy),
                   (ada_clf, ada_accuracy), (mlp_clf, mlp_accuracy)]

    return classifiers


def load_classifier(path_to_model):
    with open(path_to_model, "rb") as f:
        return pickle.load(f)
