from PIL import Image
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import os
import lxml.etree as ET
import torch
from typing import Any, Callable, Optional, Tuple
from pathlib import Path
import torchvision.transforms as transforms
from typing import Any, Callable, Optional, Tuple
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset


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
        return image,0,index


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
        return img,labels,index
    

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


class PascalVOCDataModule(TorchvisionDataset):

    def __init__(self, root, normal_class):
        super().__init__(root)

        self.root = root
        self.transformations = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        self.normal_classes = normal_class
        self.mapping = {0:'aeroplane', 1:'person', 2:'tvmonitor', 3:'dog', 4:'chair', 5:'bird', 6:'bottle', 7:'boat',
                        8:'diningtable', 9:'train', 10:'motorbike', 11:'horse', 12:'cow', 13:'bicycle', 14:'car', 15:'cat',
                        16:'sofa', 17:'bus', 18:'pottedplant', 19:'sheep'}
        

        self.image_folder = [self.mapping[normal_class] for normal_class in self.normal_classes]

        if len(self.image_folder) == 1:
            self.train_dir = f"./data_voc/{self.image_folder[0]}"
            self.train_set = FolderData(self.train_dir, transform=self.transformations)
        else:
            normal_data = []
            for folder in self.image_folder:
                self.train_dir = f"./data_voc/{folder}"
                normal_data.append(FolderData(self.train_dir, transform=self.transformations))
            self.train_set = ConcatDataset(normal_data)

        self.test_set = VOCDataset(root = './VOCSegmentation', image_set="val", transform=self.transformations, normal_classes = self.normal_classes)
        print(f"Train set size : {len(self.train_set)}")
        print(f"Test set size : {len(self.test_set)}")

