import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import random

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.data_path = os.path.join(opt.data_dir, 'images_val')
        self.label_path = os.path.join(opt.data_dir, 'labels_val')
        self.data = self.read_file(self.data_path)
        self.labels = self.read_file(self.label_path)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        img = Image.open(img)
        label = Image.open(label)

        # img, label = self.center_crop(img, label, crop_size_img, crop_size_label)

        img, label = self.img_transform(img, label)
        label = label*255
        return img, label

    def __len__(self):
        return len(self.data)
    
    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list
    
    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        p1 = random.randint(0,1)
        p2 = random.randint(0,1)
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                transforms.RandomRotation(10, resample=False, expand=False, center=None),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        seed = np.random.randint(2147483647)
        random.seed(seed)
        img = transform(img)
        label = transform(label)

        return img, label

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.data_path = os.path.join(opt.data_dir, 'images_train')
        self.label_path = os.path.join(opt.data_dir, 'labels_train')
        self.data = self.read_file(self.data_path)
        self.labels = self.read_file(self.label_path)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        img = Image.open(img)
        label = Image.open(label)

        # img, label = self.center_crop(img, label, crop_size_img, crop_size_label)

        img, label = self.img_transform(img, label)
        label = label*255
        return img, label

    def __len__(self):
        return len(self.data)
    
    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list
    
    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        p1 = random.randint(0,1)
        p2 = random.randint(0,1)
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                transforms.RandomRotation(10, resample=False, expand=False, center=None),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        seed = np.random.randint(2147483647)
        random.seed(seed)
        img = transform(img)
        label = transform(label)

        return img, label

class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.data_path = os.path.join(opt.data_dir, 'images_test')
        self.label_path = os.path.join(opt.data_dir, 'labels_test')
        self.data = self.read_file(self.data_path)
        self.labels = self.read_file(self.label_path)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        img = Image.open(img)
        label = Image.open(label)

        # img, label = self.center_crop(img, label, crop_size_img, crop_size_label)

        img, label = self.img_transform(img, label)
        label = label*255
        # print(torch.max(label))
        return img, label

    def __len__(self):
        return len(self.data)
    
    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list
    
    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        p1 = random.randint(0,1)
        p2 = random.randint(0,1)
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                transforms.RandomRotation(10, resample=False, expand=False, center=None),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        seed = np.random.randint(2147483647)
        random.seed(seed)
        img = transform(img)
        label = transform(label)

        return img, label



def get_dataset(opt):
    dataset_val = DatasetVal(opt)
    dataset_train = DatasetTrain(opt)
    dataset_test = DatasetTest(opt)

    return dataset_val, dataset_train, dataset_test