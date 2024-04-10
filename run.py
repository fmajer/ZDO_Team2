from dataset import IncisionDataset
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from methods import detect_edges, create_feature_vector
from utilities import calculate_accuracy
import torch
import torch.nn as nn

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((50, 180)),
])

incision_dataset = IncisionDataset(xml_file='data/annotations.xml',
                                   image_dir='data/',
                                   transform=data_transform)

image_id = 0
img, gray_img, thr_img, mask, n_stitches = incision_dataset.__getitem__(image_id)

# n_stitches_pred = detect_edges(gray_img)
# accuracy = calculate_accuracy(incision_dataset, detect_edges)
# print(f"Accuracy: {accuracy * 100:.2f}%")
# create_feature_vector(gray_img)

# Split dataset into training and validation
train_percentage = 0.8
train_size = int(train_percentage * incision_dataset.__len__())
val_size = incision_dataset.__len__() - train_size
train_dataset, val_dataset = random_split(incision_dataset, [train_size, val_size])

# Create dataloaders
train_batch_size = 1
val_batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size)
# print(next(iter(train_dataloader))[0].shape)
