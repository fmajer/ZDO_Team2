from load_data import load_anns_file, get_anns_dict, get_binary_mask, get_n_stitches, threshold_img
from dataset import IncisionDataset
import matplotlib.pyplot as plt
from skimage import color
from torch.utils.data import random_split, DataLoader

anns_file = load_anns_file('data/annotations.xml')
# print(anns_file["annotations"].keys())

image_id = 0
image = plt.imread('data/' + anns_file["annotations"]["image"][image_id]["@name"])
gray_img = color.rgb2gray(image)*255
thr_img = threshold_img(gray_img, 120)

anns_dict = get_anns_dict(anns_file, image_id)
mask = get_binary_mask(image, anns_dict)
n_stitches = get_n_stitches(anns_dict)

plt.imshow(image)
plt.imshow(thr_img, cmap='gray')
plt.imshow(mask, cmap='gray')
# plt.show()

incision_dataset = IncisionDataset(xml_file='data/annotations.xml',
                                   image_dir='data/')

for i, sample in enumerate(incision_dataset):
    print(i, sample[0].shape, sample[1].shape, sample[2])
    if i == 4:
        break

train_percentage = 0.8
train_size = int(train_percentage * incision_dataset.__len__())
val_size = incision_dataset.__len__() - train_size
train_dataset, val_dataset = random_split(incision_dataset, [train_size, val_size])


