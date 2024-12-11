import torch
import torchvision as tv
from abc import ABC, abstractmethod
import os
import scipy.io as sio
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, CIFAR100, StanfordCars
from torchvision.datasets.fgvc_aircraft import FGVCAircraft
import torchvision.transforms as trans
from mvtec_ad import MVTecAD
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive
from torchvision import transforms as trans
import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.datasets import CocoDetection
from torch.utils.data import Subset
import os.path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

from pycocotools.coco import COCO



def sparse_to_coarse(targets):
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]


def _convert_label(x):
    '''
    convert anomaly label. 0: normal; 1: anomaly.
    :param x (int): class label
    :return: 0 or 1
    '''
    return 0 if x == 0 else 1



class MVTecAD_Handler(VisionDataset):
    def __init__(self, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
        self.transform = trans.Compose([trans.Resize((224, 224)), trans.ToTensor()])
        self.target_transform = trans.Lambda(_convert_label)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "MVTecAD"
        self.num_classes = 2
        self.val_transform = val_transformations
        self.subset_name_list = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
        self.subset_name = self.subset_name_list[normal_classes]



    # load data
        self.train_dataset = MVTecAD('data/mvtec',
                        subset_name=self.subset_name,
                        train=True,
                        transform=transformations,
                        mask_transform=self.transform,
                        target_transform=self.target_transform,
                        download=True)
        
        self.test_dataset = MVTecAD('data/mvtec',
                        subset_name=self.subset_name,
                        train=False,
                        transform=self.val_transform,
                        mask_transform=self.transform,
                        target_transform=self.target_transform,
                        download=True)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = None
        self.val_dataset = None

    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_val_loader(self):
        return self.val_loader
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_test_dataset(self):
        return self.test_dataset
    
    def get_val_dataset(self):
        return self.val_dataset
    
    def get_num_classes(self):  
        return self.num_classes
    
    def get_dataset_name(self):
        return self.dataset_name
    
    def get_num_samples(self):
        return len(self.train_dataset)
    


class Cub2011(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cub2011, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target



class Dataset(torch.nn.Module):
    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass

    @abstractmethod
    def get_val_loader(self):
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    @abstractmethod
    def get_val_dataset(self):
        pass
    
    @abstractmethod
    def get_num_classes(self):
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass


class NormalSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, subset_indices):
        self.dataset = dataset
        self.subset_indices = subset_indices

    def __getitem__(self, index):
        return self.dataset[self.subset_indices[index]]

    def __len__(self):
        return len(self.subset_indices)


class CocoNormalSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, subset_indices):
        self.dataset = dataset
        self.subset_indices = subset_indices
    
    def __getitem__(self, index):
        label = 0 if index in self.subset_indices else 1
        return self.dataset[index], label

    def __len__(self):
        return len(self.dataset)



class Cifar100_Handler(Dataset):
    def __init__(self, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "Cifar100"
        self.num_classes = 100
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = CIFAR100(root="data/temp", train=True, download=True, transform=self.transform)
        self.train_dataset.targets = sparse_to_coarse(self.train_dataset.targets)
        self.test_dataset = CIFAR100(root="data/temp", train=False, download=True, transform=self.val_transform )
        self.test_dataset.targets = sparse_to_coarse(self.test_dataset.targets)
        self.binarize_test_labels()
        ## split the dataset to train and validation
        normal_subset_indices, normal_subset = self.get_normal_sebset_indices()
        self.normal_dataset = NormalSubsetDataset(self.train_dataset, normal_subset_indices)
        print("Normal Subset Size: ", len(self.normal_dataset))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.normal_dataset, [int(len(self.normal_dataset)*0.8), int(len(self.normal_dataset)*0.2)])
        self.train_dataset = self.normal_dataset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_val_loader(self):
        return self.val_loader

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_num_classes(self):
        return self.num_classes

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_normal_classes(self):
        return self.normal_classes

    def get_normal_sebset_indices(self):
        normal_subset_indices = [i for i, (data, label) in enumerate(self.train_dataset) if label in self.normal_classes]
        normal_subset = torch.utils.data.Subset(self.train_dataset, normal_subset_indices)
        return normal_subset_indices, normal_subset
    
    def binarize_test_labels(self):
        for i, label in enumerate(self.test_dataset.targets):
            if label in self.normal_classes:
                self.test_dataset.targets[i] = 0
            else:
                self.test_dataset.targets[i] = 1
   

class StanfordCars_Handler(Dataset):
    def __init__(self, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "StanfordCars"
        self.num_classes = 196
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = StanfordCars(root="/data/temp", split="train", download=True, transform=self.transform)
        self.test_dataset = StanfordCars(root="/data/temp", split="test", download=True, transform=self.val_transform )
        self.binarize_test_labels()
        ## split the dataset to train and validation
        normal_subset_indices, normal_subset = self.get_normal_sebset_indices()
        self.normal_dataset = NormalSubsetDataset(self.train_dataset, normal_subset_indices)
        print("Normal Subset Size: ", len(self.normal_dataset))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.normal_dataset, [int(len(self.normal_dataset)*0.8), int(len(self.normal_dataset)*0.2)])
        self.train_dataset = self.normal_dataset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_val_loader(self):
        return self.val_loader
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_test_dataset(self):
        return self.test_dataset
    
    def get_val_dataset(self):
        return self.val_dataset
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_dataset_name(self):
        return self.dataset_name
    
    def get_normal_classes(self):
        return self.normal_classes

    def get_normal_sebset_indices(self):
        normal_subset_indices = [i for i, (data, label) in enumerate(self.train_dataset) if label in self.normal_classes]
        normal_subset = torch.utils.data.Subset(self.train_dataset, normal_subset_indices)
        return normal_subset_indices, normal_subset
    
    def binarize_test_labels(self):
        for i, label in enumerate(self.test_dataset.targets):
            if label in self.normal_classes:
                self.test_dataset.targets[i] = 0
            else:
                self.test_dataset.targets[i] = 1


class Coil100_multi_object_Handler(Dataset):
    def __init__(self, batch_size, train_folder, test_folder, transformations, val_transformations, num_workers, device=None) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "Coil100_multi_object"
        self.num_classes = 100
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = ImageFolder(train_folder, transform=self.transform)
        self.test_dataset = ImageFolder(test_folder, transform=self.val_transform )
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = None
        self.val_dataset = None
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_val_loader(self):
        return self.val_loader
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_test_dataset(self):
        return self.test_dataset
    
    def get_val_dataset(self):
        return self.val_dataset
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_dataset_name(self):
        return self.dataset_name
    
    def get_num_samples(self):
        return len(self.train_dataset)
    

class Cub200_Handler(Dataset):
    def __init__(self, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "cub200"
        self.num_classes = 200
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = Cub2011(root="/data/temp", train=True, download=True, transform=self.transform)
        self.test_dataset = Cub2011(root="/data/temp", train=False, download=True, transform=self.val_transform )
        self.binarize_test_labels()
        ## split the dataset to train and validation
        normal_subset_indices, normal_subset = self.get_normal_sebset_indices()
        self.normal_dataset = NormalSubsetDataset(self.train_dataset, normal_subset_indices)
        print("Normal Subset Size: ", len(self.normal_dataset))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.normal_dataset, [int(len(self.normal_dataset)*0.8), int(len(self.normal_dataset)*0.2)])
        self.train_dataset = self.normal_dataset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_val_loader(self):
        return self.val_loader

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_num_classes(self):
        return self.num_classes

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_normal_classes(self):
        return self.normal_classes

    def get_normal_sebset_indices(self):
        normal_subset_indices = [i for i, (data, label) in enumerate(self.train_dataset) if label in self.normal_classes]
        normal_subset = torch.utils.data.Subset(self.train_dataset, normal_subset_indices)
        return normal_subset_indices, normal_subset
    
    def binarize_test_labels(self):
        ## binarize the dataframe self.test_dataset.data.targets to 0 and 1
        for i, label in enumerate(self.test_dataset.data['target']):
            if i == 2829:
                print("label: ", label)
            if label in self.normal_classes:
                self.test_dataset.data.at[i, 'target'] = 1.0
            else:
                self.test_dataset.data.at[i, 'target'] = 2.0
            
        print(self.test_dataset.data.at[2929, 'target'])
        
        print("test dataset targets: ", self.test_dataset.data['target'])

        ## check if there is any value except 1, 2 in the self.test_dataset.data.targets
        for i, label in enumerate(self.test_dataset.data['target']):
            if label != 1.0 and label != 2.0:
                print("Error: label is not 1 or 2")
                exit(1)



class MyCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False, normal_class_list=None):
        super().__init__(root, train=train, transform=transform, download=download)
        self.normal_class_list = normal_class_list
        self.normal_subset_indices = []
        for i, label in enumerate(self.targets) :
            if label in normal_class_list:
                self.normal_subset_indices.append(i)
        
    def __getitem__(self, index):
        if self.train:
            index = self.normal_subset_indices[index]
        else:
            index = index
        img, target = super().__getitem__(index)
        if target in self.normal_class_list:
            target = torch.Tensor([0])
        else:
            target = torch.Tensor([1])
        
        return img, target

    def __len__(self):
        if self.train:
            return len(self.normal_subset_indices)
        else:
            return len(self.data)




class Cifar10_Handler(Dataset):
    def __init__(self, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "Cifar10"
        self.num_classes = 10
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = MyCIFAR10(root="data/temp", train=True, download=True, transform=self.transform, normal_class_list=normal_classes)
        self.test_dataset = MyCIFAR10(root="data/temp", train=False, download=True, transform=self.val_transform, normal_class_list=normal_classes)
        ## split the dataset to train and validation
        # normal_subset_indices, normal_subset = self.get_normal_sebset_indices()
        # self.binarize_test_labels()
        # self.normal_dataset = NormalSubsetDataset(self.train_dataset, normal_subset_indices)
        self.normal_dataset = self.train_dataset
        print("Normal Subset Size: ", len(self.normal_dataset))
        # self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.normal_dataset, [int(len(self.normal_dataset)*0.8), int(len(self.normal_dataset)*0.2)])
        self.train_dataset = self.normal_dataset
        self.val_dataset = self.normal_dataset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_val_loader(self):
        return self.val_loader

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_num_classes(self):
        return self.num_classes

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_normal_classes(self):
        return self.normal_classes
    
    def get_num_samples(self):
        return len(self.train_dataset)

    def get_normal_sebset_indices(self):
        normal_classes_set = set(self.normal_classes)
        normal_subset_indices = []
        for i, (data, label) in enumerate(self.train_dataset) :
            if label in normal_classes_set:
                normal_subset_indices.append(i)
         
        # normal_subset_indices = [i for i, (data, label) in enumerate(self.train_dataset) if label in normal_classes_set]
        # normal_subset_indices = [i for i, (data, label) in enumerate(self.train_dataset) if label in self.normal_classes]
        normal_subset = torch.utils.data.Subset(self.train_dataset, normal_subset_indices)
        return normal_subset_indices, normal_subset
    
    def binarize_test_labels(self):
        for i, label in enumerate(self.test_dataset.targets):
            if label in self.normal_classes:
                self.test_dataset.targets[i] = 0
            else:
                self.test_dataset.targets[i] = 1
            

class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, normal_classes, transform=None, target_transform=None, train=True):
        super().__init__(root, annFile, transform, target_transform)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.normal_classes = normal_classes
        self.normal_subset_indices = self.filter_by_category(normal_classes)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.train == True:
            id = self.normal_subset_indices[index]
        else:
            id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transform is not None:
            image = self.transform(image)
        
        if id in self.normal_subset_indices:
            target = torch.Tensor([0])
        else:
            target = torch.Tensor([1])

        return image, target

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def filter_by_category(self, category_ids):
        subset_indices = []
        img_ids = list(self.coco.imgs.keys())  # Get all image IDs

        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=category_ids, iscrowd=None)
            annotations = self.coco.loadAnns(ann_ids)
            if len(annotations) > 0:
                subset_indices.append(self.coco.imgs[img_id]['id'])
        return subset_indices

    def __len__(self):
        if self.train == True:
            return len(self.normal_subset_indices)
        else:
            return len(self.ids)


class Coco_Handler(Dataset):
    def __init__(self, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "Coco"
        self.num_classes = 80
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = CocoDataset(root="data/coco/train2014", annFile="data/coco/annotations/instances_train2014.json", normal_classes=normal_classes, transform=transformations, train=True)
        self.test_dataset = CocoDataset(root="data/coco/val2014", annFile="data/coco/annotations/instances_val2014.json",normal_classes=normal_classes, transform=val_transformations, train=False)
        # Get the indices for the subset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = None
        self.val_dataset = None
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_val_loader(self):
        return self.val_loader
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_test_dataset(self):
        return self.test_dataset
    
    def get_val_dataset(self):
        return self.val_dataset
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_dataset_name(self):
        return self.dataset_name
    
    def get_num_samples(self):
        return len(self.train_dataset)
   
    
class FMNIST_Handler(Dataset):
    def __init__(self, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "MNIST"
        self.num_classes = 10
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = FashionMNIST(root="data/temp", train=True, download=True, transform=self.transform)
        self.test_dataset = FashionMNIST(root="data/temp", train=False, download=True, transform=self.val_transform)
        self.binarize_test_labels()
        ## split the dataset to train and validation
        normal_subset_indices, normal_subset = self.get_normal_sebset_indices()
        self.normal_dataset = NormalSubsetDataset(self.train_dataset, normal_subset_indices)
        print("Normal Subset Size: ", len(self.normal_dataset))
        # self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.normal_dataset, [int(len(self.normal_dataset)*0.8), int(len(self.normal_dataset)*0.2)])
        self.train_dataset = self.normal_dataset
        self.val_dataset = self.normal_dataset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_val_loader(self):
        return self.val_loader

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_num_classes(self):
        return self.num_classes

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_normal_classes(self):
        return self.normal_classes
    
    def get_num_samples(self):
        return len(self.train_dataset)

    def get_normal_sebset_indices(self):
        normal_classes_set = set(self.normal_classes)
        normal_subset_indices = []
        for i, (data, label) in enumerate(self.train_dataset) :
            if label in normal_classes_set:
                normal_subset_indices.append(i)
        normal_subset = torch.utils.data.Subset(self.train_dataset, normal_subset_indices)
        return normal_subset_indices, normal_subset
    
    def binarize_test_labels(self):
        for i, label in enumerate(self.test_dataset.targets):
            if label in self.normal_classes:
                self.test_dataset.targets[i] = 2
            else:
                self.test_dataset.targets[i] = 1
   


class MNIST_Handler(Dataset):
    def __init__(self, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "MNIST"
        self.num_classes = 10
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = MNIST(root="data/temp", train=True, download=True, transform=self.transform)
        self.test_dataset = MNIST(root="data/temp", train=False, download=True, transform=self.val_transform)
        self.binarize_test_labels()
        ## split the dataset to train and validation
        normal_subset_indices, normal_subset = self.get_normal_sebset_indices()
        self.normal_dataset = NormalSubsetDataset(self.train_dataset, normal_subset_indices)
        print("Normal Subset Size: ", len(self.normal_dataset))
        # self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.normal_dataset, [int(len(self.normal_dataset)*0.8), int(len(self.normal_dataset)*0.2)])
        self.train_dataset = self.normal_dataset
        self.val_dataset = self.normal_dataset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_val_loader(self):
        return self.val_loader

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_num_classes(self):
        return self.num_classes

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_normal_classes(self):
        return self.normal_classes
    
    def get_num_samples(self):
        return len(self.train_dataset)

    def get_normal_sebset_indices(self):
        normal_classes_set = set(self.normal_classes)
        normal_subset_indices = []
        for i, (data, label) in enumerate(self.train_dataset) :
            if label in normal_classes_set:
                normal_subset_indices.append(i)
        normal_subset = torch.utils.data.Subset(self.train_dataset, normal_subset_indices)
        return normal_subset_indices, normal_subset
    
    def binarize_test_labels(self):
        for i, label in enumerate(self.test_dataset.targets):
            if label in self.normal_classes:
                self.test_dataset.targets[i] = 2
            else:
                self.test_dataset.targets[i] = 1




class FGVC_Handler(Dataset):
    def __init__(self, batch_size, normal_classes, abnormal_classes, transformations, val_transformations, num_workers, device=None):
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.abnormal_classes = abnormal_classes
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "FGVC"
        self.num_classes = 100
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = FGVCAircraft(root="/data/temp", split="trainval", transform=self.transform, download=True)
        self.test_dataset = FGVCAircraft(root="/data/temp", split="test", transform=self.val_transform, download=True)
        ## split the dataset to train and validation
        normal_subset_indices, normal_subset = self.get_sebset_indices(self.train_dataset, self.normal_classes)
        all_test_indices, all_test = self.get_sebset_indices(self.test_dataset, self.normal_classes + self.abnormal_classes)
        self.binarize_test_labels()
        self.normal_dataset = NormalSubsetDataset(self.train_dataset, normal_subset_indices)
        self.test_dataset = NormalSubsetDataset(self.test_dataset, all_test_indices)
        print("Normal Subset Size: ", len(self.normal_dataset))
        # self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.normal_dataset, [int(len(self.normal_dataset)*0.8), int(len(self.normal_dataset)*0.2)])
        self.train_dataset = self.normal_dataset
        self.val_dataset = self.normal_dataset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_val_loader(self):
        return self.val_loader

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_num_classes(self):
        return self.num_classes

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_normal_classes(self):
        return self.normal_classes

    def get_sebset_indices(self, dataset, subset_classes):
        normal_subset_indices = [i for i, (data, label) in enumerate(dataset) if label in subset_classes]
        normal_subset = torch.utils.data.Subset(dataset, normal_subset_indices)
        return normal_subset_indices, normal_subset
    
    def binarize_test_labels(self):
        for i, label in enumerate(self.test_dataset._labels):
            if label in self.normal_classes:
                self.test_dataset._labels[i] = 0
            else:
                self.test_dataset._labels[i] = 1

## create a main to test the code
if __name__ == "__main__":
    transformations = tv.transforms.Compose([tv.transforms.Resize((224, 224)), tv.transforms.ToTensor()])
    x = torch.randn(3, 32, 32)
    # data_handler = MVTecAD_Handler(16, "screw", transformations, 4, "cuda:6")
    data_handler = Coco_Handler(16, [7], transformations, transformations, 8, "cuda:6")
    trainset = data_handler.get_train_dataset()
    print("train set size: ", len(trainset))
    train_loader = data_handler.get_train_loader()
    # for i, (x, y) in enumerate(train_loader):
    #     print(x.shape)
    #     print(y.shape)
    #     print(y)
    #     break

    testset = data_handler.get_test_dataset()
    print("test set size: ", len(testset))
    test_loader = data_handler.get_test_loader()
    for i, (x, y) in enumerate(test_loader):
        ## save  the images 
        fig = plt.figure()
        for j in range(16):
            ax = fig.add_subplot(4, 4, j+1)
            ax.imshow(x[j].permute(1, 2, 0))
            ax.set_title(y[j].item())
            ax.axis("off")
        plt.savefig("test_images.png")
        
        # break




