import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MyDataSet(Dataset):
    """Custom dataset supporting FR and NR modes."""

    def __init__(self, data_frame, root_dir, mask_dir, transform=None, test_mode=1):
        self.mos_data = data_frame
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.test_mode = test_mode

        # Recursively collect image files; support relative paths and plain filenames.
        self.image_files = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                rel_path = os.path.relpath(os.path.join(dirpath, f), root_dir)
                self.image_files[rel_path.lower()] = rel_path
                self.image_files[f.lower()] = rel_path

        # Recursively collect mask files.
        self.mask_files = {}
        for dirpath, _, filenames in os.walk(mask_dir):
            for f in filenames:
                rel_path = os.path.relpath(os.path.join(dirpath, f), mask_dir)
                self.mask_files[rel_path.lower()] = rel_path
                self.mask_files[f.lower()] = rel_path
        if self.test_mode == 0:
            self.dataset_config = None
        else:
            self.dataset_config = None

        self.current_index = 0

    def get_image_path(self, image_name):
        return os.path.join(self.root_dir, image_name)

    def extract_prefix(self, filename, is_mask=False):
        """Extract prefix by dataset naming rule.
           - `is_mask=True`: always split by `.`
           - `is_mask=False`: split by `dataset_config["prefix_split"]`
        """
        if is_mask:
            return filename.split('.')[0]
        else:
            return filename.split(self.dataset_config["prefix_split"])[0]

    def build_fr_mask_map(self):
        """Automatically match reference masks for distorted images."""
        ref_mask_map = {}
        for mask_name in self.mask_files.keys():
            ref_prefix = self.extract_prefix(mask_name, is_mask=True)

            for img_name in self.image_files.keys():
                img_prefix = self.extract_prefix(img_name, is_mask=False)
                if img_prefix == ref_prefix:
                    ref_mask_map[img_name] = self.mask_files[mask_name]

        return ref_mask_map

    def __len__(self):
        return len(self.mos_data)

    def __getitem__(self, idx):
        img_name = self.mos_data.iloc[idx]['Image'].lower()
        if img_name not in self.image_files:
            raise FileNotFoundError(f"Image {img_name} not found in {self.root_dir}")

        img_path = os.path.join(self.root_dir, self.image_files[img_name])

        if self.test_mode == 1:
            if img_name not in self.mask_files:
                raise FileNotFoundError(f"Mask {img_name} not found in {self.mask_dir}")
            mask_path = os.path.join(self.mask_dir, self.mask_files[img_name])
        else:
            ref_name = self.mos_data.iloc[idx]['hq_Image'].lower()
            if ref_name not in self.mask_files:
                raise FileNotFoundError(f"Reference mask {ref_name} not found in {self.mask_dir}")
            mask_path = os.path.join(self.mask_dir, self.mask_files[ref_name])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform["image"](image)
            mask = self.transform["mask"](mask)

        mos = torch.tensor(self.mos_data.iloc[idx]['MOS'], dtype=torch.float32)
        self.current_index += 1
        return image, mask, mos

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batched tensors."""
        images, masks, mos_scores = zip(*batch)
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        mos_scores = torch.stack(mos_scores, dim=0)
        return images, masks, mos_scores
