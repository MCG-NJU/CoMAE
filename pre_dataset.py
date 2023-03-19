import os.path
import random

import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile

from torchvision.datasets.folder import make_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch

from torchvision.transforms import functional as F
import torchvision.datasets as datasets


class AlignedConcDataset(datasets.DatasetFolder):

    def __init__(self, data_dir=None, transform=None, labeled=True):
        self.transform = transform
        self.data_dir = data_dir
        self.labeled = labeled
        self.target_transform=None
        #self.transforms = StandardTransform(transform, target_transform)

        if labeled:
            self.classes = [d.name for d in os.scandir(self.data_dir) if d.is_dir()]
            self.classes.sort()
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
            self.imgs = make_dataset(self.data_dir, self.class_to_idx, 'png')
            self.samples=self.imgs
            self.root=self.data_dir
            self.targets = [s[1] for s in self.samples]


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            img_path = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc = Image.open(img_path).convert('RGB')

        # split RGB and Depth as A and B
        w, h = AB_conc.size
        w2 = int(w / 2)
        A = AB_conc.crop((0, 0, w2, h))
        B = AB_conc.crop((w2, 0, w, h))
        if self.labeled:
            sample = {'A': A, 'B': B, 'img_name': img_name, 'label': label}
        else:
            sample = {'A': A, 'B': B, 'img_name': img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomCrop(transforms.RandomCrop):
    def __call__(self, sample):
        w, h = sample.size
        w2 = int(w / 2)
        A = sample.crop((0, 0, w2, h))
        B = sample.crop((w2, 0, w, h))
        if self.padding :
            A = F.pad(A, self.padding)
            B = F.pad(B, self.padding)
        # pad the width if needed
        if self.pad_if_needed and A.size[0] < self.size[1]:
            A = F.pad(A, (int((1 + self.size[1] - A.size[0]) / 2), 0))
            B = F.pad(B, (int((1 + self.size[1] - B.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and A.size[1] < self.size[0]:
            A = F.pad(A, (0, int((1 + self.size[0] - A.size[1]) / 2)))
            B = F.pad(B, (0, int((1 + self.size[0] - B.size[1]) / 2)))
        i, j, h, w = self.get_params(A, self.size)
        A = F.crop(A, i, j, h, w)
        B = F.crop(B, i, j, h, w)

        target_images = Image.new('RGB', (448, 224))
        target_images.paste(A, (0, 0, 224, 224))
        target_images.paste(B, (224, 0, 448, 224))

        return target_images

class RandomErasing(transforms.RandomErasing):
    def __call__(self, sample):
        A=sample[:, :, :224]
        B=sample[:, :, 224:]
        x, y, h, w, v = self.get_params(A, scale=self.scale, ratio=self.ratio, value=self.value)
        A=F.erase(A, x, y, h, w, v, self.inplace)
        B= F.erase(B, x, y, h, w, v, self.inplace)


        target_images=torch.cat((A, B), dim=2)

        return target_images



class RandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, sample):
        w, h = sample.size
        w2 = int(w / 2)
        A = sample.crop((0, 0, w2, h))
        B = sample.crop((w2, 0, w, h))
        i, j, h, w = self.get_params(A, self.scale, self.ratio)
        A=F.resized_crop(A, i, j, h, w, self.size, self.interpolation)
        B=F.resized_crop(B, i, j, h, w, self.size, self.interpolation)
        target_images = Image.new('RGB', (448, 224))
        target_images.paste(A, (0, 0, 224, 224))
        target_images.paste(B, (224, 0, 448, 224))
        return target_images



class ColorJitter(transforms.ColorJitter):
    def __call__(self, sample):
        w, h = sample.size
        w2 = int(w / 2)
        A = sample.crop((0, 0, w2, h))
        B = sample.crop((w2, 0, w, h))

        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                A = F.adjust_brightness(A, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                A = F.adjust_contrast(A, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                A = F.adjust_saturation(A, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                A = F.adjust_hue(A, hue_factor)

        target_images = Image.new('RGB', (448, 224))
        target_images.paste(A, (0, 0, 224, 224))
        target_images.paste(B, (224, 0, 448, 224))
        return target_images


class CenterCrop(transforms.CenterCrop):
    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        sample['A'] = F.center_crop(A, self.size)
        sample['B'] = F.center_crop(B, self.size)
        return sample


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, sample):
        w, h = sample.size
        w2 = int(w / 2)
        A = sample.crop((0, 0, w2, h))
        B = sample.crop((w2, 0, w, h))
        if random.random() > 0.5:
            A = F.hflip(A)
            B = F.hflip(B)
        target_images = Image.new('RGB', (448, 224))
        target_images.paste(A, (0, 0, 224, 224))
        target_images.paste(B, (224, 0, 448, 224))
        return target_images


class Resize(transforms.Resize):
    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        h = self.size[0]
        w = self.size[1]
        sample['A'] = F.resize(A, (h, w))
        sample['B'] = F.resize(B, (h, w))
        return sample


