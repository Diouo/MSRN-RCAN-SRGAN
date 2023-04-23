from os import listdir
from os.path import join
from PIL import Image

from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop
import torch.utils.data as data


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform():
    return Compose([
        ToTensor(),
    ])


def augment_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
    ])


def get_training_set(upscale_factor,crop_size, dataSet='DIV2K'):

    root_dir = "/home/guozy/BISHE/dataset/" + dataSet + "/images/"
    train_dir = join(root_dir, "train")
    valid_crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(valid_crop_size, upscale_factor),
                             target_transform=target_transform(),
                             augment_transform=augment_transform(valid_crop_size)
                             )


def get_test_set(upscale_factor,crop_size, dataSet='DIV2K'):

    root_dir = "/home/guozy/BISHE/dataset/" + dataSet + "/images/"
    test_dir = join(root_dir, "test")
    valid_crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(valid_crop_size, upscale_factor),
                             target_transform=target_transform(),
                             augment_transform=augment_transform(valid_crop_size)
                             )


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    #y, _, _ = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None, augment_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment_transform = augment_transform

    # only random crop
    def __getitem__(self, index):
        input_image = load_img(self.image_filenames[index])

        input_image = self.augment_transform(input_image) # random crop
        target = input_image.copy()
        input_image = self.input_transform(input_image) # resize and totensor
        target = self.target_transform(target) # totensor

        return input_image, target # LR & HR

    def __len__(self):
        return len(self.image_filenames)
