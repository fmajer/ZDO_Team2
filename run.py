from dataset import IncisionDataset
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from edge_detection import detect_edges
from utilities import calculate_accuracy

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((50, 180)),
])

incision_dataset = IncisionDataset(xml_file='data/annotations.xml',
                                   image_dir='data/',
                                   transform=None)

image_id = 5
img, gray_img, thr_img, mask, n_stitches = incision_dataset.__getitem__(image_id)

plt.imshow(gray_img, cmap='gray')
plt.imshow(thr_img, cmap='gray')

n_stitches_pred = detect_edges(gray_img)

accuracy = calculate_accuracy(incision_dataset, detect_edges)
print(f"Accuracy: {accuracy * 100:.2f}%")











# -------------------------------------------------USELESS (for now)----------------------------------------------------

# Split dataset into training and validation
train_percentage = 0.8
train_size = int(train_percentage * incision_dataset.__len__())
val_size = incision_dataset.__len__() - train_size
train_dataset, val_dataset = random_split(incision_dataset, [train_size, val_size])

# Create dataloaders
batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
# print(next(iter(train_dataloader))[0].shape)
