import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import skimage
from skimage.morphology import skeletonize
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


def test_2(image):
    grayscale = skimage.color.rgb2gray(image)
    diameter = 2
    template = (255 * np.ones([4 * diameter, 4 * diameter])).astype("uint8")
    rr, cc = skimage.draw.disk((template.shape[0] / 2, template.shape[1] / 2), diameter / 2)
    template[rr, cc] = 0

    # template = (np.zeros([20, 3])).astype("uint8")
    # template[:, 1] = 255

    result = skimage.feature.match_template(grayscale, template)

    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    plt.figure()
    plt.imshow(grayscale)
    plt.figure()
    plt.imshow(template)
    plt.figure()
    plt.imshow(result)

    result[result < 0.3] = 0
    plt.figure()
    plt.imshow(result)

    plt.show()


def test_1(image):
    image = (255 * skimage.color.gray2rgb(skimage.color.rgb2gray(image))).astype(np.uint8)
    # plt.figure()
    # plt.imshow(image)

    color_quantized_image = color_quantization(image, 3)
    color = color_quantized_image[0]
    count = np.sum(np.concatenate(np.all(color_quantized_image == color, axis=-1)))
    original_img = image

    # plt.figure()
    # for index, i in enumerate(np.linspace(0, 2, 50)):

    #     color_quantized_image = color_quantization(image, 3)
    #     color = color_quantized_image[0]
    #     new_count = np.sum(np.concatenate(np.all(color_quantized_image == color, axis=-1)))

    #     print(count, "->", new_count)
    #     count = new_count
    #     plt.subplot(5, 10, index + 1)
    #     plt.imshow(color_quantized_image)
    #     image = skimage.exposure.adjust_gamma(original_img, i)

    # plt.show()
    # plt.figure()
    # plt.imshow(color_quantized_image)
    color_with_most_lines_as_img = color_with_most_lines(color_quantized_image)
    # plt.figure()
    # plt.imshow(color_with_most_lines_as_img)

    # plt.show()
    colors = np.unique(color_with_most_lines_as_img)
    assert len(colors) == 2

    b = skimage.morphology.area_closing(color_with_most_lines_as_img, 5, 2)
    # edges = skimage.feature.canny(b, 2, 1, 25)

    skelet = skeletonize(b)

    lines = skimage.transform.probabilistic_hough_line(skelet, threshold=5, line_length=5, line_gap=3)
    gray = np.zeros(color_with_most_lines_as_img.shape, dtype=np.uint8)

    for line in lines:
        p0, p1 = line
        cv2.line(gray, (p0[0], p0[1]), (p1[0], p1[1]), 255)

    # skelet = np.array((skimage.morphology.skeletonize(b) * 255), dtype="uint8")
    # minLineLength = 15
    # lines = cv2.HoughLinesP(image=skelet, rho=0.2, theta=np.pi / 360, threshold=10, lines=np.array([]), minLineLength=minLineLength, maxLineGap=80)

    # if lines is not None:
    #     a, b, c = lines.shape
    #     print(lines.shape)
    #     for i in range(a):
    #         cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255), 1, cv2.LINE_AA)

    return (image, color_with_most_lines_as_img, b, skelet, gray)


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


def plot_a_lot_of_images(incision_dataset):
    data = [test_2(incision_dataset.__getitem__(i)[0]) for i in range(8 * 10)]
    data = [test_1(incision_dataset.__getitem__(i)[0]) for i in range(8 * 10)]
    plot_images([d[0] for d in data], 8, 10)
    plot_images([d[1] for d in data], 8, 10)
    plot_images([d[2] for d in data], 8, 10)
    plot_images([d[3] for d in data], 8, 10)
    plot_images([d[4] for d in data], 8, 10)
    plt.show()

    # plot_images([preprocess_image(incision_dataset.__getitem__(i)[0])[1] for i in range(8*10)], 8, 10)
    a = incision_dataset.__getitem__(0)[0].cpu().numpy()
    a = (a, 0, -1)
    plot_images([np.moveaxis(incision_dataset.__getitem__(i)[0].cpu().numpy(), 0, -1) for i in range(8 * 10)], 8, 10)
    plt.show()


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
