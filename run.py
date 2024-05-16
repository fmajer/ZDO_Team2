from glob import glob
from typing import Callable
from dataset import IncisionDataset
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from hog_lbp_classification import train_lbp_hog_classifier, get_hog_lbp_classifier_accuracy
from neural_network import train_nn, load_nn
from hog_classification import train_hog_classifier, get_hog_classifier_accuracy, hog_predict
from lbp_classification import train_lbp_classifier, get_lbp_classifier_accuracy
from utilities import calculate_accuracy, load_classifier
from edge_detector import detect_edges, hough_vert_edge_detect
import random
import copy
import numpy as np
import matplotlib.pyplot as plt


# Define data transformations
data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(50, 180)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.RandomAutocontrast(),
        transforms.ColorJitter(brightness=0.2, contrast=0.3, hue=0.1),
    ]
)

# Parameters
train_nn_param = False
train_classifiers = False
check_accuracy = False

# Create dataset
incision_dataset = IncisionDataset(xml_file="data/annotations.xml", image_dir="data/",
                                   transform=data_transform if train_nn_param else None)

# Split dataset into training and validation
# seed = random.randint(0, 10000)
seed = 100
train_percentage = 0.9
generator = torch.Generator().manual_seed(seed)
train_size = int(train_percentage * incision_dataset.__len__())
val_size = incision_dataset.__len__() - train_size
train_dataset, val_dataset = random_split(incision_dataset, [train_size, val_size], generator=generator)

# Create dataloaders
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Get data sample with specified id
image_id = 5
img, mask, n_stitches = incision_dataset.__getitem__(image_id)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train, check accuracy or predict for given images
if train_nn_param:
    n_epochs = 500
    learning_rate = 0.001
    path_to_nn_func: Callable[[float], str] = lambda x: f"trained_models/{x:.2f}_neural_network_seed_{seed}.pth"  # noqa: E731
    train_nn(n_epochs, learning_rate, incision_dataset, train_dataloader,
             val_dataloader, train_size, val_size, device, path_to_nn_func)

elif train_classifiers:
    train_hog_classifier(train_dataset, val_dataset, "trained_models/trained_hog_classifier.pkl")
    train_lbp_classifier(train_dataset, val_dataset, "trained_models/trained_lbp_classifier.pkl")
    train_lbp_hog_classifier(train_dataset, val_dataset, "trained_models/trained_hog_lbp_classifier.pkl")

elif check_accuracy:
    path_to_nn = glob(f"trained_models/*_neural_network_seed_{seed}.pth")[0]
    path_to_hog = "trained_models/trained_hog_classifier.pkl"
    path_to_lbp = "trained_models/trained_lbp_classifier.pkl"
    path_to_hog_lbp = "trained_models/trained_hog_lbp_classifier.pkl"
    nn = load_nn(path_to_nn)
    hog = load_classifier(path_to_hog)
    lbp = load_classifier(path_to_lbp)
    hog_lbp = load_classifier(path_to_hog_lbp)

    print(f"Edge detector - accuracy: {calculate_accuracy(val_dataset, detect_edges) * 100:.2f}%")
    print(f"Neural network - accuracy: {calculate_accuracy(val_dataset, nn.predict) * 100:.2f}%")
    print(f"HOG classifier - accuracy: {get_hog_classifier_accuracy(val_dataset, hog) * 100:.2f}%")
    print(f"LBP classifier - accuracy: {get_lbp_classifier_accuracy(val_dataset, lbp) * 100:.2f}%")
    print(f"HOG+LBP classifier - accuracy: {get_hog_lbp_classifier_accuracy(val_dataset, hog_lbp) * 100:.2f}%")

else:
    # plt.imshow(img.squeeze().permute(1, 2, 0).numpy())
    plt.imshow(img)
    plt.show()

    path_to_nn = glob(f"trained_models/*_neural_network_seed_{seed}.pth")[0]
    path_to_hog = "trained_models/trained_hog_classifier.pkl"
    nn = load_nn(path_to_nn)
    hog = load_classifier(path_to_hog)

    nn_stitches_pred = nn.predict(img)
    hog_stitches_pred = hog_predict(img, hog)
    print(f"Neural network prediction: {nn_stitches_pred} stitches")
    print(f"HOG classifier prediction: {hog_stitches_pred} stitches")
    print(f"Edge detector prediction: {detect_edges(copy.deepcopy(img))} stitches")
    # hough_vert_edge_detect(img)
