from os import listdir
from os.path import join
from PIL import Image
import random
from typing import Sequence

import torch
from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, RandomVerticalFlip
import torch.utils.data as data
from torchvision.utils import _log_api_usage_once
import torchvision.transforms.functional as ttf

class MyRotateTransform(torch.nn.Module):
    def __init__(self, angles: Sequence[int]):
        super().__init__()
        _log_api_usage_once(self)
        self.angles = angles

    def forward(self, img):
        angle = random.choice(self.angles)
        return ttf.rotate(img, angle * 90)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angles={self.angles})"


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def augment_transform(crop_size, mode):

    if mode == 'train':
        return Compose([RandomCrop(crop_size), RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), MyRotateTransform([0,1,2,3])])
    elif mode == 'test':
        return Compose([CenterCrop(crop_size),])


def input_transform(crop_size, upscale_factor):
    return Compose([
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform():
    return Compose([
        ToTensor(),
    ])


def get_training_set(upscale_factor,crop_size, dataSet='DIV2K'):

    root_dir = "/home/guozy/BISHE/dataset/" + dataSet + "/images/"
    train_dir = join(root_dir, "train")
    valid_crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(train_dir,
                             augment_transform=augment_transform(valid_crop_size, 'train'),
                             input_transform=input_transform(valid_crop_size, upscale_factor),
                             target_transform=target_transform(), 
                             )


def get_test_set(upscale_factor,crop_size, dataSet='DIV2K'):

    if dataSet == 'DIV2K':
        root_dir = "/home/guozy/BISHE/dataset/" + dataSet + "/images/"
        test_dir = join(root_dir, "test")
    elif dataSet == 'BSD100':
        test_dir = '/home/guozy/BISHE/dataset/BSD100/'
    
    valid_crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(test_dir,
                             augment_transform=augment_transform(valid_crop_size, 'test'),
                             input_transform=input_transform(valid_crop_size, upscale_factor),
                             target_transform=target_transform(),
                             )


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    #y, _, _ = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,augment_transform=None, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.augment_transform = augment_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_image = load_img(self.image_filenames[index])

        input_image = self.augment_transform(input_image) # random crop for train; center crop for test;
        target = input_image.copy()
        input_image = self.input_transform(input_image) # resize and totensor
        target = self.target_transform(target)

        return input_image, target # LR & HR

    def __len__(self):
        return len(self.image_filenames)
