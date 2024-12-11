from typing import Any, Callable, Optional, Tuple
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
import lxml.etree as ET
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO


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

    def __init__(self, batch_size, normal_classes, transformations,  dir="./VOCSegmentation", num_workers=2) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dir = dir
        self.transformations = transformations
        self.normal_classes = normal_classes
        self.mapping = {0:'aeroplane', 1:'person', 2:'tvmonitor', 3:'dog', 4:'chair', 5:'bird', 6:'bottle', 7:'boat',
                        8:'diningtable', 9:'train', 10:'motorbike', 11:'horse', 12:'cow', 13:'bicycle', 14:'car', 15:'cat',
                        16:'sofa', 17:'bus', 18:'pottedplant', 19:'sheep'}
        

        self.image_folder = [self.mapping[normal_class] for normal_class in self.normal_classes]

        if len(self.image_folder) == 1:
            self.train_dir = f"./data_voc/{self.image_folder[0]}"
            self.train_dataset = FolderData(self.train_dir, transform=self.transformations)
        else:
            normal_data = []
            for folder in self.image_folder:
                self.train_dir = f"./data_voc/{folder}"
                normal_data.append(FolderData(self.train_dir, transform=self.transformations))
            self.train_dataset = ConcatDataset(normal_data)

        self.val_dataset = VOCDataset(self.dir, image_set="val", transform=self.transformations, normal_classes = self.normal_classes)
        self.test_dataset = VOCDataset(self.dir, image_set="val", transform=self.transformations, normal_classes = self.normal_classes)
        print(f"Train set size : {len(self.train_dataset)}")
        print(f"Val set size : {len(self.val_dataset)}")
        print(f"Test set size : {len(self.test_dataset)}")


    def get_train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def get_val_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_test_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_val_dataset_size(self):
        return len(self.val_dataset)

    def get_test_dataset_size(self):
        return len(self.test_dataset)

    def get_module_name(self):
        return "PascalVOCDataModule"
    
    def get_num_classes(self):
        return 20


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
        self.targets = []
        for i in range(len(self.ids)):
            if self.ids[i] in self.normal_subset_indices:
                self.targets.append(0)
            else:
                self.targets.append(1)
        

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
    def __init__(self, batch_size, normal_classes, transformations, num_workers):
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.num_workers = num_workers
        self.dataset_name = "Coco"
        self.num_classes = 80
        self.transform = transformations
        self.train_dataset = CocoDataset(root="./train2014", annFile="./annotations/instances_train2014.json", normal_classes=normal_classes, transform=transformations, train=True)
        self.test_dataset = CocoDataset(root="./val2014", annFile="./annotations/instances_val2014.json", normal_classes=normal_classes, transform=transformations, train=False)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        print(f"Train set size : {len(self.train_dataset)}")
        print(f"Test set size : {len(self.test_dataset)}")

    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_test_dataset(self):
        return self.test_dataset
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_dataset_name(self):
        return self.dataset_name
    