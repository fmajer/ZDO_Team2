from skimage.feature import hog
from skimage.transform import resize
import matplotlib.pyplot as plt
import skimage
import pickle
from utilities import train_classifiers, load_classifier, color_quantization, color_with_most_lines
import numpy as np
# https://github.com/BilalxAI/comaprison-of-CNN-and-HOG/blob/main/cnn%20vs%20hog.ipynb


def train_hog_classifier(train_dataset, val_dataset, path_to_model):
    x_train_hog, y_train, x_val_hog, y_val = get_hog_features(train_dataset, val_dataset)

    # Get best classifier based on accuracy and get its accuracy
    classifiers = train_classifiers(x_train_hog, y_train, x_val_hog, y_val)
    best_classifier = max(classifiers, key=lambda x: x[1])
    print(f"\nBest classifier: {best_classifier[0]} with accuracy: {best_classifier[1]}\n")

    # Print all classifiers and their accuracies ordered by accuracy
    classifiers.sort(key=lambda x: x[1], reverse=True)
    for classifier in classifiers:
        print(f"Classifier: {classifier[0]} with accuracy: {classifier[1]}")

    # Save best classifier
    with open(path_to_model, "wb") as f:
        pickle.dump(best_classifier[0], f)


def get_hog_classifier_accuracy(val_dataset, path_to_model):
    x_val_hog, y_val = get_hog_features(val_dataset, val_dataset)[2:]

    classifier = load_classifier(path_to_model)

    return classifier.score(x_val_hog, y_val)


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
