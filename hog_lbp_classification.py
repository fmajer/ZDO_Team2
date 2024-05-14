import pickle
import numpy as np
from hog_classification import get_hog_features
from lbp_classification import LocalBinaryPatterns, get_lbp_features
from utilities import train_classifiers, load_classifier


def train_lbp_hog_classifier(train_dataset, val_dataset, path_to_model):
    x_train_hog, y_train, x_val_hog, y_val = get_hog_features(train_dataset, val_dataset)
    x_train_lbp, y_train, x_val_lbp, y_val = get_lbp_features(LocalBinaryPatterns(50, 5), train_dataset, val_dataset)
    x_train, y_train, x_val, y_val = combine_hog_and_lbp_features(x_train_hog, x_val_hog, x_train_lbp, x_val_lbp, y_train, y_val)

    # Get best classifier based on accuracy and get its accuracy
    classifiers = train_classifiers(x_train, y_train, x_val, y_val)

    best_classifier = max(classifiers, key=lambda x: x[1])
    print(f"\nBest classifier: {best_classifier[0]} with accuracy: {best_classifier[1]}\n")

    # Print all classifiers and their accuracies ordered by accuracy
    classifiers.sort(key=lambda x: x[1], reverse=True)
    for classifier in classifiers:
        print(f"Classifier: {classifier[0]} with accuracy: {classifier[1]}")

    # Save best classifier
    with open(path_to_model, "wb") as f:
        pickle.dump(best_classifier[0], f)


def get_lbp_hog_classifier_accuracy(val_dataset, path_to_model):
    x_val_hog, y_val = get_hog_features(val_dataset, val_dataset)[2:]

    classifier = load_classifier(path_to_model)

    return classifier.score(x_val_hog, y_val)


def combine_hog_and_lbp_features(x_train_hog, x_val_hog, x_train_lbp, x_val_lbp, y_train, y_val):
    x_train = np.concatenate((x_train_hog, x_train_lbp), axis=1)
    x_val = np.concatenate((x_val_hog, x_val_lbp), axis=1)
    return x_train, y_train, x_val, y_val