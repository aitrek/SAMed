import os
import random

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
import json
from torchvision import transforms


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


class MyDataset(Synapse_dataset):

    def __init__(self, data_root: str, split: str, test_smaple_rate: float,
                 transform=None, excluded=None, min_mask_region_area: int = 10):
        self.data_root = data_root
        self.split = split
        self.test_smaple_rate = test_smaple_rate
        self.transform = transform
        self.min_mask_region_area = min_mask_region_area
        self.excluded = excluded if excluded else []
        self.data_list, self.label_list = self._read_data()

    def _read_data(self):
        data_paths = []
        label_paths = []
        for name in os.listdir(self.data_root):
            if name in self.excluded:
                print(f"skip excluded dataset: {name}")
                continue

            if name.startswith(".") or name.startswith("__"):
                continue

            dataset_dir = os.path.join(self.data_root, name)
            if not os.path.isdir(dataset_dir):
                continue
            print(f"load {self.split} dataset: {dataset_dir}")
            split_path = os.path.join(
                dataset_dir,
                f"split_seed-{42}_test_size-{0.1}.json"
            )
            if not os.path.exists(split_path):
                print(f"split file not exists: {split_path}")
                continue

            with open(split_path) as f:
                split_data = json.load(f)

            if "train" in self.split.lower():
                data_list = split_data["train"]
            else:
                data_list = random.choices(
                    split_data["test"],
                    k=int(len(split_data["test"]) * self.test_smaple_rate)
                )

            for data_path, label_path in data_list:
                data_path = os.path.join(dataset_dir, data_path)
                label_path = os.path.join(dataset_dir, label_path)

                mask = np.load(label_path)
                mask_vals = [
                    _ for _ in np.unique(mask)
                    if _ != 0 and (mask == _).sum() > self.min_mask_region_area
                ]
                if not mask_vals:
                    continue
                data_paths.append(data_path)
                label_paths.append(label_path)
                # for mask_val in mask_vals:
                #     data_paths.append(data_path)
                #     label_paths.append((label_path, mask_val))

        return data_paths, label_paths

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        label_path = self.label_list[idx]
        image = cv2.imread(data_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = np.load(label_path)
        label[label != 0] = 1
        case_name = os.path.basename(data_path)
        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = case_name
        return sample
