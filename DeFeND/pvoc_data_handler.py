from PIL import Image
import lxml.etree as ET
import os
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import torchvision as tv

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

class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, split,transforms,normal_classes):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.segmentation_dir = os.path.join(root_dir, 'SegmentationClass')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')
        self.image_ids = []        
        self.transform = transforms
        self.normal_classes = normal_classes
        self.split = split
        if split == 'train':
            self.split_list_path = os.path.join(root_dir, 'ImageSets/Segmentation/train.txt')
        elif split == 'val':
            self.split_list_path = os.path.join(root_dir, 'ImageSets/Segmentation/val.txt')
        else:
            raise Exception('Choose between "train" and "val"')
        f = open(self.split_list_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.image_ids.append(line)
        f.close()
        self.label_mapping = {'aeroplane':0, 'person':1, 'tvmonitor':2, 'dog':3, 'chair':4, 'bird':5, 'bottle':6, 'boat':7, 
                 'diningtable':8, 'train':9, 'motorbike':10, 'horse':11, 'cow':12, 'bicycle':13, 'car':14, 'cat':15, 
                 'sofa':16, 'bus':17, 'pottedplant':18, 'sheep':19}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, filename + '.jpg')
        annotation_path = os.path.join(self.annotation_dir, filename + '.xml')        
        segmentation_path = os.path.join(self.segmentation_dir, filename + '.png')
        image = Image.open(image_path).convert('RGB').resize((224,224)) # Read image 
        annotation = self._parse_annotation(annotation_path) # Read label 
        if self.split == 'val':
            annotation = 0 if annotation in self.normal_classes else 1
        else:
            pass
        segmentation = Image.open(segmentation_path).resize((224,224)) # Read segmentation map 
        if segmentation.mode != "RGB":
            segmentation = segmentation.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
            segmentation = self.transform(segmentation)
        return image,annotation 

    def _parse_annotation(self, annotation_path:str) -> list:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text 
            labels.append(label)
        main_label = labels[0]
        number_label = self.label_mapping.get(main_label)
        return number_label

class NormalSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, subset_indices):
        self.dataset = dataset
        self.subset_indices = subset_indices

    def __getitem__(self, index):
        return self.dataset[self.subset_indices[index]]

    def __len__(self):
        return len(self.subset_indices)

class PVOC_handler(Dataset):
    def __init__(self, root_dir, batch_size, normal_classes, transformations, val_transformations, num_workers, device=None):
        super(PVOC_handler, self).__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.normal_classes = normal_classes
        self.num_workers = num_workers
        self.device = device
        self.dataset_name = "Pascal VOC"
        self.num_classes = 20
        self.transform = transformations
        self.val_transform = val_transformations
        self.train_dataset = PascalVOCDataset(root_dir, 'trainval', self.transform, normal_classes)
        self.test_dataset = PascalVOCDataset(root_dir, 'val', self.val_transform, normal_classes)
        normal_subset_indices, _ = self.get_normal_sebset_indices()
        self.normal_dataset = NormalSubsetDataset(self.train_dataset, normal_subset_indices)
        print("Normal Subset Size: ", len(self.normal_dataset))
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


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
    
    def get_normal_classes(self):
        return self.normal_classes

    def get_normal_sebset_indices(self):
        normal_subset_indices = [i for i, (img, label) in enumerate(self.train_dataset) if label in self.normal_classes]
        normal_subset = torch.utils.data.Subset(self.train_dataset, normal_subset_indices)
        return normal_subset_indices, normal_subset


if __name__ == "__main__":
    transformations = tv.transforms.Compose([tv.transforms.Resize((224, 224)), tv.transforms.ToTensor()])
    pvoc_handler = PVOC_handler(root_dir='VOC2012/', batch_size=32, normal_classes=[0, 1], transformations=transformations, val_transformations=transformations, num_workers=1, device="cpu")
    trainset = pvoc_handler.get_train_dataset()
    print("Train set size: ", len(trainset))
    train_loader = pvoc_handler.get_train_loader()
    for i, (x, y) in enumerate(train_loader):
        print(x.shape)
        print(y.shape)
        break

    testset = pvoc_handler.get_test_dataset()
    print("test set size: ", len(testset))
    test_loader = pvoc_handler.get_test_loader()
    for i, (x, y) in enumerate(test_loader):
        print(x.shape)
        print(y.shape)
        print(y)
        break
