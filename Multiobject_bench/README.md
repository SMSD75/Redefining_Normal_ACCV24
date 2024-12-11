# Multi-object Anomaly Detection Benchmark

This project aims to provide a benchmark for several anomaly segmentation methods in the field of computer vision, specifically adapted for multi-object scenarios. We have modified the original implementations by redesigning their data loaders and incorporating multi-object training and evaluation capabilities. This adaptation allows for a fair comparison of these methods in more complex, real-world settings where multiple objects may be present. The benchmark includes several state-of-the-art repositories, including:

1. KDAD ([paper](https://arxiv.org/abs/2011.11108) | [original repo](https://github.com/rohban-lab/Knowledge_Distillation_AD))
2. RD4AD ([paper](https://arxiv.org/abs/2201.10703) | [original repo](https://github.com/hq-deng/RD4AD))
3. Puzzle_AD ([paper](https://arxiv.org/pdf/2008.12959.pdf) | [original repo](https://github.com/Niousha12/Puzzle_Anomaly_Detection))
4. Mean-Shifted AD ([paper](https://arxiv.org/pdf/2106.03844.pdf) | [original repo](https://github.com/talreiss/Mean-Shifted-Anomaly-Detection))
5. Transformaly ([paper](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Cohen_Transformaly_-_Two_Feature_Spaces_Are_Better_Than_One_CVPRW_2022_paper.pdf) | [original repo](https://github.com/MatanCohen1/Transformaly))
6. Deep-SVDD ([paper](http://proceedings.mlr.press/v80/ruff18a.html) | [original repo](https://github.com/lukasruff/Deep-SVDD-PyTorch))
7. GeneralAD ([paper](https://arxiv.org/abs/2407.12427) | [original repo](https://github.com/LucStrater/GeneralAD))

## Supported Datasets
The benchmark currently supports a set of datasets, and efforts are ongoing to include more diverse datasets over time. 

### Current Supported Datasets:
- MNIST
- FMNIST
- CIFAR10
- CIFAR100
- MVTec
- Pascal VOC
- COCO

*(List will be updated as more datasets are incorporated.)*

## Evaluation ##

The following table shows the AUROC scores of the single-class, one-vs-all, setting. * shows results reproduced by us. CSI and FITYMI will be added.

|                | RDAD         | CSI  | FITYMI  | KDAD         | Puzzle AD    | MSAD         | Transformaly        | Deep SVDD  | General AD   | MKD+         |
|----------------|--------------|-------------------|-----------------------|--------------|--------------|--------------|----------------------|------------|--------------|--------------|
| **Backbone**   | WideRes-50   | ResNet-18        | ViT-B16-224           | VGG-16       | U-Net        | ViT-B16-224  | ViT-B16-384/224     | LeNet      | ViT-B14-518  | ViT-B16-224  |
| **Pre-training** | Supervised | -           | Supervised            | Supervised   | -   | Supervised   | Supervised         | - | Self-supervised   | Supervised   |
| **CIFAR10**    | 86.1         | 94.3             | 99.1                  | 87.2         | 72.47        | 97.2         | 98.3/94.9*          | 64.81      | 99.1         | 98.6         |
| **CIFAR100**   | -            | 89.6             | 98.1                  | 80.6         | -            | 96.4         | 97.3/93*            | -          | 98           | 97.4         |
| **FMNIST**     | 95           | 94.2             | 80.5*                 | 94.5         | 92.6         | 94.2         | 94.4/92.7*          | -          | 94.6         | 94.4         |
| **Pascal VOC** | 58.6         | -                | -                     | 82.8*        | 55*          | 91.8*        | 82.5*               | 56.14*     | 93.4*       | 95.4         |
| **COCO Detection** | 47.9     | -                | -                     | 75.4*        | -            | 86.7*        | 75.4*               | -          | 90.5*            | 94.5         |


## Getting Started
To get started with the benchmark, follow the instructions in the respective repository folders. Use the requirements.txt or the env.yml files to create the environment.

## License
This benchmark is distributed under the [MIT License](LICENSE). Please review the license file before using or contributing to the repository.
