from torch.utils.data import Dataset
from load_data import load_anns_file, get_anns_dict, get_binary_mask, get_n_stitches
import matplotlib.pyplot as plt


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
        return int(self.anns_file["annotations"]['meta']['task']['size'])

    def __getitem__(self, idx):
        img = plt.imread(self.image_dir + self.anns_file["annotations"]["image"][idx]["@name"])
        anns_dict = get_anns_dict(self.anns_file, idx)
        mask = get_binary_mask(img, anns_dict)
        n_stitches = get_n_stitches(anns_dict)

        if self.transform:
            img = self.transform(img)

        return img, mask, n_stitches

