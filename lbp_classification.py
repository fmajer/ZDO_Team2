import pickle
from skimage import feature
from skimage.transform import resize
import numpy as np
import cv2
from utilities import train_classifiers


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


def train_lbp_classifier(train_dataset, val_dataset, path_to_model):
    radius = 5
    n_points = 10 * radius
    desc = LocalBinaryPatterns(n_points, radius)

    # Get LBP features for training and validation datasets
    x_train_lbp, y_train, x_val_lbp, y_val = get_lbp_features(desc, train_dataset, val_dataset)

    # Get best classifier based on accuracy and get its accuracy
    classifiers = train_classifiers(x_train_lbp, y_train, x_val_lbp, y_val)
    best_classifier = max(classifiers, key=lambda x: x[1])
    print(f"\nBest classifier: {best_classifier[0]} with accuracy: {best_classifier[1]}\n")

    # Print all classifiers and their accuracies ordered by accuracy
    classifiers.sort(key=lambda x: x[1], reverse=True)
    for classifier in classifiers:
        print(f"Classifier: {classifier[0]} with accuracy: {classifier[1]}")

    # Save best classifier
    with open(path_to_model, "wb") as f:
        pickle.dump(best_classifier[0], f)


def get_lbp_classifier_accuracy(val_dataset, classifier):
    x_val_lbp, y_val = get_lbp_features(LocalBinaryPatterns(50, 5), None, val_dataset)[2:]
    return classifier.score(x_val_lbp, y_val)


def get_lbp_features(desc, train_dataset, val_dataset):
    x_train_lbp, y_train = [], []
    x_val_lbp, y_val = [], []

    if train_dataset is not None:
        for i in range(len(train_dataset)):
            img, _, n_stitches = train_dataset.__getitem__(i)
            resized_img = resize(img, (50, 160)).astype('uint8') * 255
            gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
            hist = desc.describe(gray)

            x_train_lbp.append(hist)
            y_train.append(n_stitches)

    for i in range(len(val_dataset)):
        img, _, n_stitches = val_dataset.__getitem__(i)
        resized_img = resize(img, (50, 160)).astype('uint8') * 255
        gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
        hist = desc.describe(gray)

        x_val_lbp.append(hist)
        y_val.append(n_stitches)

    return x_train_lbp, y_train, x_val_lbp, y_val
