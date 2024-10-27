import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import config
import utils


class CustomImageDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = config.DATASET_PATH + utils.get_img_path(self.data.iloc[idx]["image"])
        image = Image.open(img_path).convert("RGB")
        label = self.__get_label__(idx)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __get_label__(self, idx):
        return utils.to_label(self.data.iloc[idx]["kp-1"])

class TestSet(Dataset):
    
    def __init__(self, transform=None):
        self.data = pd.read_csv(config.TEST_SET_PATH)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = config.DATASET_PATH + utils.get_img_path(self.data.iloc[idx]["image"])
        print(img_path)
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label
    


class CustomPatchDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data['metadata'] = self.data['kp-1'].apply(utils.to_coordinates_and_labels)
        self.data = self.data.explode('metadata')
        self.data = self.data[['image', 'metadata']]
        self.data.dropna(inplace=True)
    
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = config.DATASET_PATH + utils.get_img_path(self.data.iloc[idx]["image"])
        image = Image.open(img_path).convert("RGB")
        x, y, label = self.data.iloc[idx]["metadata"]

        image = utils.extract_patch(image, x, y, config.PATCH_SIZE)
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def __get_label__(self, idx):
        return utils.patch_to_label(self.data.iloc[idx]["kp-1"])

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, mode='classification', batch_size=32, num_workers=4):
        super().__init__()

        if mode not in ['classification', 'patch']:
            raise ValueError("Mode must be either 'classification' or 'patch'")

        self.mode = mode
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define image transformations
        self.train_transforms = transforms.Compose([
            transforms.Lambda(utils.normalize_brightness_PIL),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.Lambda(utils.normalize_brightness_PIL),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):

        full_dataset = None
        self.test_set = TestSet(transform=self.val_transforms)

        if self.mode == 'classification':
            full_dataset = CustomImageDataset(csv_file=self.csv_file, transform=self.train_transforms)

        elif self.mode == 'patch':
            full_dataset = CustomPatchDataset(csv_file=self.csv_file, transform=self.train_transforms)

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        self.val_dataset.dataset.transform = self.val_transforms

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)