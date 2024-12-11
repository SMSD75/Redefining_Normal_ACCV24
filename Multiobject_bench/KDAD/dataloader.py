import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CIFAR100
from torchvision.datasets import ImageFolder
from PIL import Image
import lxml.etree as ET
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset

def sparse2coarse(targets):
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

class FolderData(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.image_paths = [os.path.join(root, filename) for filename in os.listdir(root) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image,0


class VOCDataset(Dataset):
    def __init__(
            self,
            root: str,
            image_set: str = "trainaug",
            normal_classes: list = None,
            transform: Optional[Callable] = None,
    ):
        super(VOCDataset, self).__init__()
        self.transform = transform
        self.normal_classes = normal_classes
        self.image_set = image_set
        if self.image_set == "trainaug" or self.image_set == "train":
            seg_folder = "SegmentationClassAug"
        elif self.image_set == "val":
            seg_folder = "SegmentationClass"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, 'images')
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir) or not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(root, 'sets')
        split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        self.annot_dir = os.path.join(root, 'Annotations')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(seg_dir, x + ".png") for x in file_names]
        self.test_annotations = [os.path.join(self.annot_dir, x + ".xml") for x in file_names]

        assert all([Path(f).is_file() for f in self.masks]) and all([Path(f).is_file() for f in self.images])

        self.str2ing_mapping = {'aeroplane':0, 'person':1, 'tvmonitor':2, 'dog':3, 'chair':4, 'bird':5, 'bottle':6, 'boat':7, 
            'diningtable':8, 'train':9, 'motorbike':10, 'horse':11, 'cow':12, 'bicycle':13, 'car':14, 'cat':15, 
            'sofa':16, 'bus':17, 'pottedplant':18, 'sheep':19}
        
        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                         'train', 'tvmonitor']
        
        self.proper_class_names = ['aeroplane', 'person', 'tvmonitor', 'dog', 'chair', 'bird', 'bottle', 'boat', 
                                   'diningtable', 'train', 'motorbike', 'horse', 'cow', 'bicycle', 'car', 'cat',
                                   'sofa', 'bus', 'pottedplant', 'sheep']

        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img = Image.open(self.images[index]).convert('RGB')

        if self.image_set == "trainaug" or self.image_set == "train":
                
            mask = Image.open(self.masks[index])
            mask = transforms.functional.to_tensor(mask)
            mask = mask * 255
            mask = mask.to(torch.int64)
            labels = torch.unique(mask)
            labels = labels[(labels != 0) & (labels != 255)]
            labels = labels -1
            labels = [self.class_names[label.item()] for label in labels]
            labels = [self.str2ing_mapping[label] for label in labels]

        elif self.image_set == "val":
            annotation_path = self.test_annotations[index]
            labels = self._parse_annotation(annotation_path) 
            if self.normal_classes is not None:
                if any(x in self.normal_classes for x in labels):
                    labels = 0
                else:
                    labels = 1
        if self.transform:
            img = self.transform(img)
        return img,labels
    

    def __len__(self) -> int:
        return len(self.images)
    
    def _parse_annotation(self, annotation_path:str) -> list:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text 
            labels.append(label)
        labels = [self.str2ing_mapping[label] for label in labels]
        return labels
    
    def save_images_to_class_folders(self, output_dir):
        for i in range(len(self.images)):
            img = Image.open(self.images[i])
            labels = self.__getitem__(i)[1]  

            for label in labels:
                label_folder = os.path.join(output_dir, self.proper_class_names[label])
                os.makedirs(label_folder, exist_ok=True)
                img_name = os.path.basename(self.images[i])
                img_path = os.path.join(label_folder, img_name)
                img.save(img_path)

                for other_label in labels:
                    if other_label != label:
                        other_label_folder = os.path.join(output_dir, self.proper_class_names[other_label])
                        os.makedirs(other_label_folder, exist_ok=True)
                        img_copy_path = os.path.join(other_label_folder, img_name)
                        img.save(img_copy_path)


class PascalVOCDataModule():

    def __init__(self, batch_size, normal_classes, train_transform, test_transform,  img_dir="./VOCSegmentation", num_workers=2) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dir = img_dir
        self.image_train_transform = train_transform
        self.image_test_transform = test_transform
        self.normal_classes = normal_classes
        self.mapping = {0:'aeroplane', 1:'person', 2:'tvmonitor', 3:'dog', 4:'chair', 5:'bird', 6:'bottle', 7:'boat',
                        8:'diningtable', 9:'train', 10:'motorbike', 11:'horse', 12:'cow', 13:'bicycle', 14:'car', 15:'cat',
                        16:'sofa', 17:'bus', 18:'pottedplant', 19:'sheep'}
        

        self.image_folder = [self.mapping[normal_class] for normal_class in self.normal_classes]

        if len(self.image_folder) == 1:
            self.train_dir = f"./data_voc/{self.image_folder[0]}"
            self.train_dataset = FolderData(self.train_dir, transform=self.image_train_transform)
        else:
            normal_data = []
            for folder in self.image_folder:
                self.train_dir = f"./data_voc/{folder}"
                normal_data.append(FolderData(self.train_dir, transform=self.image_train_transform))
            self.train_dataset = ConcatDataset(normal_data)

        self.test_dataset = VOCDataset(self.dir, image_set="val", transform=self.image_test_transform, normal_classes = self.normal_classes)
        print(f"Train set size : {len(self.train_dataset)}")
        print(f"Test set size : {len(self.test_dataset)}")

    def get_train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def get_test_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=16, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_test_dataset_size(self):
        return len(self.test_dataset)

    def get_module_name(self):
        return "PascalVOCDataModule"
    
    def get_num_classes(self):
        return 20
    


def load_data(config):
    normal_class = config['normal_class']
    batch_size = config['batch_size']

    if config['dataset_name'] not in ['pvoc']:

        if config['dataset_name'] in ['cifar10']:
            img_transform = transforms.Compose([
                transforms.Resize((256, 256), Image.Resampling.LANCZOS),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            os.makedirs("./KDAD_Dataset/CIFAR10/train", exist_ok=True)
            dataset = CIFAR10('./KDAD_Dataset/CIFAR10/train', train=True, download=True, transform=img_transform)
            print("Cifar10 DataLoader Called...")
            print("All Train Data: ", dataset.data.shape)
            dataset.data = dataset.data[np.isin(dataset.targets, normal_class)]
            print("Normal Train Data: ", dataset.data.shape)

            os.makedirs("./KDAD_Dataset/CIFAR10/test", exist_ok=True)
            test_set = CIFAR10("./KDAD_Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)
            print("Test Train Data:", test_set.data.shape)


        elif config['dataset_name'] in ['cifar100']:
            img_transform = transforms.Compose([
                transforms.Resize((256, 256), Image.Resampling.LANCZOS),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            os.makedirs("./KDAD_Dataset/CIFAR100/train", exist_ok=True)
            dataset = CIFAR100('./KDAD_Dataset/CIFAR100/train', train=True, download=True, transform=img_transform)
            print("Cifar100 DataLoader Called...")
            print("All Train Data: ", dataset.data.shape)
            dataset.targets = sparse2coarse(dataset.targets)
            dataset.data = dataset.data[np.isin(dataset.targets, normal_class)]
            print("Normal Train Data: ", dataset.data.shape)

            os.makedirs("./KDAD_Dataset/CIFAR10/test", exist_ok=True)
            test_set = CIFAR100("./KDAD_Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)
            test_set.targets = sparse2coarse(test_set.targets)
            print("Test Train Data:", test_set.data.shape)


        elif config['dataset_name'] in ['mnist']:
            img_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToTensor()
            ])
            os.makedirs("./KDAD_Dataset/MNIST/train", exist_ok=True)
            dataset = MNIST('./KDAD_Dataset/MNIST/train', train=True, download=True, transform=img_transform)
            print("MNIST DataLoader Called...")
            print("All Train Data: ", dataset.data.shape)
            dataset.data = dataset.data[np.isin(dataset.targets, normal_class)]
            print("Normal Train Data: ", dataset.data.shape)

            os.makedirs("./KDAD_Dataset/MNIST/test", exist_ok=True)
            test_set = MNIST("./KDAD_Dataset/MNIST/test", train=False, download=True, transform=img_transform)
            print("Test Train Data:", test_set.data.shape)


        elif config['dataset_name'] in ['fashionmnist']:
            img_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),  
                transforms.ToTensor()
            ])
            os.makedirs("./KDAD_Dataset/FashionMNIST/train", exist_ok=True)
            dataset = FashionMNIST('./KDAD_Dataset/FashionMNIST/train', train=True, download=True, transform=img_transform)
            print("FashionMNIST DataLoader Called...")
            print("All Train Data: ", dataset.data.shape)
            dataset.data = dataset.data[np.isin(dataset.targets, normal_class)]
            print("Normal Train Data: ", dataset.data.shape)

            os.makedirs("./KDAD_Dataset/FashionMNIST/test", exist_ok=True)
            test_set = FashionMNIST("./KDAD_Dataset/FashionMNIST/test", train=False, download=True, transform=img_transform)
            print("Test Train Data:", test_set.data.shape)


        elif config['dataset_name'] in ['mvtec']:
            data_path = 'KDAD_Dataset/MVTec/' + normal_class + '/train'
            mvtec_img_size = config['mvtec_img_size']
            orig_transform = transforms.Compose([
                transforms.Resize([mvtec_img_size, mvtec_img_size]),
                transforms.ToTensor()
            ])
            dataset = ImageFolder(root=data_path, transform=orig_transform)
            test_data_path = 'KDAD_Dataset/MVTec/' + normal_class + '/test'
            test_set = ImageFolder(root=test_data_path, transform=orig_transform)



        elif config['dataset_name'] in ['retina']:
            data_path = 'KDAD_Dataset/OCT2017/train'
            orig_transform = transforms.Compose([
                transforms.Resize([128, 128]),
                transforms.ToTensor()
            ])
            dataset = ImageFolder(root=data_path, transform=orig_transform)
            test_data_path = 'KDAD_Dataset/OCT2017/test'
            test_set = ImageFolder(root=test_data_path, transform=orig_transform)

        else:
            raise Exception(
                "You enter {} as dataset, which is not a valid dataset for this repository!".format(config['dataset_name']))

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
        )
    
    else:
        
        if type(normal_class) != list:
            normal_class = [normal_class]
            
        img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        mod = PascalVOCDataModule(batch_size = batch_size, train_transform = img_transform, test_transform = img_transform, normal_classes = normal_class)
        train_dataloader = mod.get_train_dataloader()
        test_dataloader = mod.get_test_dataloader()

    return train_dataloader, test_dataloader




def load_localization_data(config):
    normal_class = config['normal_class']
    mvtec_img_size = config['mvtec_img_size']

    orig_transform = transforms.Compose([
        transforms.Resize([mvtec_img_size, mvtec_img_size]),
        transforms.ToTensor()
    ])

    test_data_path = 'Dataset/MVTec/' + normal_class + '/test'
    test_set = ImageFolder(root=test_data_path, transform=orig_transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
    )

    ground_data_path = 'Dataset/MVTec/' + normal_class + '/ground_truth'
    ground_dataset = ImageFolder(root=ground_data_path, transform=orig_transform)
    ground_dataloader = torch.utils.data.DataLoader(
        ground_dataset,
        batch_size=512,
        num_workers=0,
        shuffle=False
    )

    x_ground = next(iter(ground_dataloader))[0].numpy()
    ground_temp = x_ground

    std_groud_temp = np.transpose(ground_temp, (0, 2, 3, 1))
    x_ground = std_groud_temp

    return test_dataloader, x_ground
