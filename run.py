from dataset import IncisionDataset
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from methods import detect_edges, create_feature_vector, detect_edges_2, random_lul
from neural_network import detect_with_nn, ConvNet
from utilities import calculate_accuracy, calculate_accuracy_2
import numpy as np

# Define data transformations
data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((50, 180)),
    ]
)

# Create dataset
incision_dataset = IncisionDataset(xml_file="data/annotations.xml", image_dir="data/", transform=data_transform)

# Split dataset into training and validation
train_percentage = 0.9
train_size = int(train_percentage * incision_dataset.__len__())
val_size = incision_dataset.__len__() - train_size
train_dataset, val_dataset = random_split(incision_dataset, [train_size, val_size])

# Create dataloaders
batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
# print(next(iter(train_dataloader))[0].shape)

# Get data sample with specified id
image_id = 0
img, gray_img, thr_img, quantized_mask, mask, n_stitches = incision_dataset.__getitem__(image_id)
# plt.imshow(img, cmap='gray')

# Detect number of stitches
detect_with_nn(incision_dataset, train_dataloader, val_dataloader, train_size, val_size)


# Březina things
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

# n_stitches_pred = detect_edges(gray_img)
# accuracy = calculate_accuracy(incision_dataset, detect_edges)
# accuracy = calculate_accuracy_2(incision_dataset, detect_edges_2)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# create_feature_vector(gray_img)



