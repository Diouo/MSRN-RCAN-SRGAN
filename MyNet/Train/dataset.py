import tarfile
from os import remove,listdir
from os.path import exists, join, basename
from PIL import Image
from six.moves import urllib

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torch.utils.data as data


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor,crop_size, dataSet='DIV2K'):

    root_dir = "/home/guozy/BISHE/dataset/" + dataSet + "/images/"
    train_dir = join(root_dir, "train")
    valid_crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(valid_crop_size, upscale_factor),
                             target_transform=target_transform(valid_crop_size))


def get_test_set(upscale_factor,crop_size, dataSet='DIV2K'):

    root_dir = "/home/guozy/BISHE/dataset/" + dataSet + "/images/"
    test_dir = join(root_dir, "test")
    valid_crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(valid_crop_size, upscale_factor),
                             target_transform=target_transform(valid_crop_size))


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    #y, _, _ = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    # attention: if crop is used, it must be a centercrop and not a random crop
    def __getitem__(self, index):
        input_image = load_img(self.image_filenames[index])
        target = input_image.copy()
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return len(self.image_filenames)
