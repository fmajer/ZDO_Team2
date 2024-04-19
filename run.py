import copy
from glob import glob
import random
from typing import Callable
from dataset import IncisionDataset
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from methods import detect_edges, create_feature_vector, detect_edges_2, random_lul
import matplotlib.pyplot as plt
from neural_network import train_nn, load_nn
from utilities import calculate_accuracy, plot_images, test_1, test_2, plot_a_lot_of_images
from utilities import color_quantization, color_with_most_lines
import numpy as np
from hog_classification import train_hog_classifier, plot_img_and_hog

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

train = False

# Create dataset
incision_dataset = IncisionDataset(xml_file="data/annotations.xml", image_dir="data/",
                                   transform=data_transform if train else None)

# Plot some images
# plot_a_lot_of_images(incision_dataset)

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
# print(next(iter(train_dataloader))[0].shape)

# Get data sample with specified id
image_id = 2
img, mask, n_stitches = incision_dataset.__getitem__(image_id)

# Train or load neural network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if train:
    path_to_nn_func: Callable[[float], str] = lambda x: f"trained_nn/{x:.2f}_neural_network_seed_{seed}.pth"  # noqa: E731
    train_nn(incision_dataset, train_dataloader, val_dataloader, train_size, val_size, device, path_to_nn_func)
else:
    path_to_nn = glob(f"trained_nn/*_neural_network_seed_{seed}.pth")[0]
    nn = load_nn(path_to_nn)

    # Try various detection methods
    nn_stitches_pred = nn.predict(img)
    edges_stitches_pred = detect_edges(copy.deepcopy(img))
    print(f"Edge detector - accuracy: {calculate_accuracy(incision_dataset, detect_edges) * 100:.2f}%")
    print(f"Neural network - accuracy: {calculate_accuracy(incision_dataset, nn.predict) * 100:.2f}%")

    print(f"HoG + Classifier - accuracy: {train_hog_classifier(train_dataset, val_dataset) * 100:.2f}%")


# Březina things

# quantized_mask = color_quantization(img, 3)
# quantized_mask = color_with_most_lines(quantized_mask)

# step = 12*8
# ids = range(0, step)

# data = [incision_dataset.__getitem__(id) for id in ids]


# for i in range(0, len(ids), step):
#     plt.figure()
#     for d, j in zip(data[i : i + step], range(0, step)):
#         plt.subplot(12, 8, j + 1)
#         plt.imshow(d[1])

#             # plt.subplot(2, 1, 2)
#             # plt.imshow(d[3])

#     plt.figure()
#     for d, j in zip(data[i : i + step], range(0, step)):
#         plt.subplot(12, 8, j + 1)
#         plt.imshow(d[2])

# plt.show()
