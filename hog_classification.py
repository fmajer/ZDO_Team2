from skimage.feature import hog
from skimage.transform import resize
import matplotlib.pyplot as plt
import skimage
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from utilities import color_quantization, color_with_most_lines
import numpy as np
# https://github.com/BilalxAI/comaprison-of-CNN-and-HOG/blob/main/cnn%20vs%20hog.ipynb


def train_hog_classifier(train_dataset, val_dataset):
    x_train_hog, y_train, x_val_hog, y_val = get_hog_features(train_dataset, val_dataset)

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

    # Get best classifier based on accuracy and get its accuracy
    classifiers = [(svm_classifier, svm_accuracy), (gbc_clf, gbc_accuracy), (kn_classifier, kn_accuracy),
                      (ada_clf, ada_accuracy), (mlp_clf, mlp_accuracy)]
    best_classifier = max(classifiers, key=lambda x: x[1])
    print(f"\nBest classifier: {best_classifier[0]} with accuracy: {best_classifier[1]}\n")

    # Print all classifiers and their accuracies ordered by accuracy
    classifiers.sort(key=lambda x: x[1], reverse=True)
    for classifier in classifiers:
        print(f"Classifier: {classifier[0]} with accuracy: {classifier[1]}")

    return best_classifier[1]


# noinspection DuplicatedCode
def get_hog_features(train_dataset, val_dataset):
    x_train_hog, y_train = [], []
    x_val_hog, y_val = [], []

    for i in range(len(train_dataset)):
        img, _, n_stitches = train_dataset.__getitem__(i)
        # quantized_mask = color_quantization(img, 3)
        # quantized_mask = color_with_most_lines(quantized_mask)
        # resized_img = resize(np.expand_dims(quantized_mask, -1), (50, 160)) # use with SVM
        resized_img = resize(img, (50, 160))
        features = hog(resized_img, orientations=9, pixels_per_cell=(2, 2),
                       cells_per_block=(2, 2), visualize=True, channel_axis=-1)
        # print(features[0].shape)
        x_train_hog.append(features[0])
        y_train.append(n_stitches)

    for i in range(len(val_dataset)):
        img, _, n_stitches = val_dataset.__getitem__(i)
        # quantized_mask = color_quantization(img, 3)
        # quantized_mask = color_with_most_lines(quantized_mask)
        # resized_img = resize(np.expand_dims(quantized_mask, -1), (50, 160)) # use with SVM
        resized_img = resize(img, (50, 160))
        features = hog(resized_img, orientations=9, pixels_per_cell=(2, 2),
                       cells_per_block=(2, 2), visualize=True, channel_axis=-1)
        x_val_hog.append(features[0])
        y_val.append(n_stitches)

    # plot_img_and_hog(resized_img, features[1])
    return x_train_hog, y_train, x_val_hog, y_val


def plot_img_and_hog(img, hog_img):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))

    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = skimage.exposure.rescale_intensity(hog_img, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
