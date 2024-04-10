from load_data import load_anns_file, get_anns_dict, get_binary_mask, get_n_stitches
from utilities import threshold_at_cumulative_value, color_quantization, color_with_most_lines
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import color


class IncisionDataset(Dataset):
    """Incision dataset."""

    def __init__(self, xml_file, image_dir, transform=None):
        """
        Arguments:
            xml_file (string): Path to the xml file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.anns_file = load_anns_file(xml_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return int(self.anns_file["annotations"]["meta"]["task"]["size"])

    def __getitem__(self, idx):
        img = np.array(plt.imread(self.image_dir + self.anns_file["annotations"]["image"][idx]["@name"]))
        # gray_img = color.rgb2gray(img) * 255
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thr_img = threshold_at_cumulative_value(gray_img, 0.01)

        # quantized_mask = color_quantization(img, 3)
        # quantized_mask = color_with_most_lines(quantized_mask)
        quantized_mask = img

        anns_dict = get_anns_dict(self.anns_file, idx)
        mask = get_binary_mask(img, anns_dict)
        n_stitches = get_n_stitches(anns_dict)

        if self.transform:
            img = self.transform(img)
            gray_img = self.transform(gray_img)
            thr_img = self.transform(thr_img)
            quantized_mask = self.transform(quantized_mask)
            mask = self.transform(mask)

        return img, gray_img, thr_img, quantized_mask, mask, n_stitches
