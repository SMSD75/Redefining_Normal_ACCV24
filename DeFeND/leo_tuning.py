from matplotlib import patches, pyplot as plt
import torch
import argparse
import os
from sqlite3 import Time
import time
import torch
from torchvision import transforms
import torch.nn.functional as F
from data_handler import Cifar100_Handler, Cifar10_Handler, Coco_Handler, Coil100_multi_object_Handler, FMNIST_Handler, MNIST_Handler, MVTecAD_Handler
from main import MKDTrainer, TransformerMKD, MaskedMKD
from models import CrossAttentionBlock, FeatureExtractor
import wandb
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from optimizer import PatchCorrespondenceOptimizer
import torchvision.transforms as trn
from torch import distributed as dist
from data import PascalVOCDataModule
from leopart import Leopart
from transformations import DenseTransforms, LeopartTransforms
from transformers import ViTModel
from timm.models.vision_transformer import vit_small_patch16_224, vit_small_patch8_224, vit_base_patch16_384

from vision_transformers import vit_base, vit_small, mae_vit_base_patch16
import numpy as np
from typing import Optional, List, Tuple, Dict
import timm



def get_backbone_prefix(weights: Dict[str, torch.Tensor], arch: str) -> Optional[Tuple[int, str]]:
    # Determine weight prefix if returns empty string as prefix if not existent.
    if 'vit' in arch:
        search_suffix = 'cls_token'
    elif 'resnet' in arch:
        search_suffix = 'conv1.weight'
    else:
        raise ValueError()
    for k in weights:
        if k.endswith(search_suffix):
            prefix_idx = len(k) - len(search_suffix)
            return prefix_idx, k[:prefix_idx]
        

def get_backbone_weights(arch: str, method: str, patch_size: int = None,
                         weight_prefix: Optional[str]= "model", ckpt_path: str = None) -> Dict[str, torch.Tensor]:
    """
    Load backbone weights into formatted state dict given arch, method and patch size as identifiers.
    :param arch: Target architecture. Currently supports resnet50, vit-small and vit-base.
    :param method: Method identifier.
    :param patch_size: Patch size of ViT. Ignored if arch is not ViT.
    :param weight_prefix: Optional prefix of weights to match model naming.
    :param ckpt_path: Optional path to checkpoint containing state_dict to be processed.
    :return: Dictionary mapping to weight Tensors.
    """
    def identity_transform(x): return x
    arch_to_args = {
        'vit-small16-dino': ("https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth",
                             torch.hub.load_state_dict_from_url,
                             lambda x: x["teacher"]),
        'vit-small16-mocov3': ("https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar",
                               torch.hub.load_state_dict_from_url,
                               lambda x: {k: v for k, v in x.items() if k.startswith('module.base_encoder')}),
        'vit-small16-sup_vit': ('vit_small_patch16_224',
                            lambda x: timm.create_model(x,  pretrained=True).state_dict(),
                            identity_transform),
        'vit-base16-sup_vit': ('vit_base_patch16_224',
                            lambda x: timm.create_model(x,  pretrained=True).state_dict(),
                            identity_transform),
        'vit-base16-dino': ("https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth",
                            torch.hub.load_state_dict_from_url,
                            lambda x: x["teacher"]),
        'vit-base8-dino': ("https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth",
                           torch.hub.load_state_dict_from_url,
                           lambda x: x["teacher"]),
        'vit-base16-mae': ("https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
                           torch.hub.load_state_dict_from_url,
                           lambda x: x["model"]),
        'resnet50-maskcontrast': (ckpt_path,
                                  torch.load,
                                  lambda  x: x["model"]),
        'resnet50-swav': ("https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
                          torch.hub.load_state_dict_from_url,
                          identity_transform),
        'resnet50-moco': ("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
                          torch.hub.load_state_dict_from_url,
                          identity_transform),
        'resnet50-densecl': (ckpt_path,
                             torch.load,
                             identity_transform),
    }
    # arch_to_args['vit-base16-ours'] = arch_to_args['vit-base8-ours'] = arch_to_args['vit-small16-ours']

    if "vit" in arch:
        url, loader, weight_transform = arch_to_args[f"{arch}{patch_size}-{method}"]
    else:
        url, loader, weight_transform = arch_to_args[f"{arch}-{method}"]
    weights = loader(url)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    weights = weight_transform(weights)
    prefix_idx, prefix = get_backbone_prefix(weights, arch)
    if weight_prefix:
        return {f"{weight_prefix}.{k[prefix_idx:]}": v for k, v in weights.items() if k.startswith(prefix)
                and "head" not in k and "prototypes" not in k}
    return {f"{k[prefix_idx:]}": v for k, v in weights.items() if k.startswith(prefix)
            and "head" not in k and "prototypes" not in k}



# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)

## set the seed for numpy operations
np.random.seed(42)


project_name = "Dense_Anomaly_Detection"
## generate ListeColorMap of distinct colors

## what are the colors for red, blue, green, brown, yello, orange, purple, white, black, maroon, olive, teal, navy, gray, silver
## Fill the ListedColormap with the colors above

# cmap = ListedColormap(['#FF0000', '#0000FF', '#008000', '#A52A2A', '#FFFF00', '#FFA500', '#800080', '#FFFFFF', '#000000', '#800000', '#808000', '#008080', '#000080', '#808080', '#C0C0C0'])




train_config = {
    "use_teacher": True,
    "roi_align_kernel_size": 7,
    "momentum_teacher": 0.9995,
    "momentum_teacher_end": 1.,
    "exclude_norm_bias": True,
    "arch": "vit-base",
    "patch_size": 16,
    "pretrained_weights": "sup_vit",  # null is represented as None in Python
    "projection_feat_dim": 256,
    "projection_hidden_dim": 2048,
    "n_layers_projection_head": 3,
    "queue_length": 8192,
    "loss_mask": "all",
    "batch_size": 32,
    "max_epochs": 1000,
    "nmb_prototypes": 5,
    "temperature": 0.1,
    "sinkhorn_iterations": 3,
    "crops_for_assign": [0, 1],
    "optimizer": "adamw",
    "lr_backbone": 0.00001,
    "lr_heads": 0.0001,
    "final_lr": 0.,
    "weight_decay": 0.04,
    "weight_decay_end": 0.5,
    "epsilon": 0.05,
    "fast_dev_run": False,
    "num_clusters_kmeans_miou": [500, 300, 21],
    "val_downsample_masks": True,
    "val_iters": 10,
    "save_checkpoint_every_n_epochs": 5,
    "checkpoint_dir": "<your ckpt dir>",
    "checkpoint": None,  # null is represented as None in Python
    "only_load_weights": True
}


class LeopartTuningTrainer():
    def __init__(self, dataloader, test_train_loader, test_dataloader, leopart_model, num_epochs, device, logger):
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.leopart_model = leopart_model
        self.device = device
        # self.leopart_model = self.leopart_model.to(self.device)
        self.test_train_loader = test_train_loader
        self.optimizer = None
        self.num_epochs = num_epochs
        self.logger = logger
        # self.logger.watch(dense_tuning_model, log="all", log_freq=10)
    
    
    def setup_optimizer(self, optimization_config):
        self.leopart_model.configure_optimizers()
    

    def train_one_epoch(self):
        self.leopart_model.train()
        epoch_loss = 0
        before_loading_time = time.time()
        self.leopart_model.on_train_epoch_start()
        for i, (batch, target) in enumerate(self.dataloader):
            ## loop on the batch items and send them to the device
            for j, data in enumerate(batch[0]):
                batch[0][j] = data.to(self.device)
            batch[1]["gc"] = batch[1]["gc"].to(self.device)
            batch[1]["all"] = batch[1]["all"].to(self.device)
            clustering_loss = self.leopart_model.training_step(batch)
            total_loss = clustering_loss
            epoch_loss += total_loss.item()
            print("Iteration: {} Loss: {}".format(i, total_loss.item()))
            self.logger.log({"clustering_loss": clustering_loss.item()})
            # lr = self.optimizer.get_lr()
            # self.logger.log({"lr": lr})
            before_loading_time = time.time()
        epoch_loss /= (i + 1)
        print("Epoch Loss: {}".format(epoch_loss))


    def validate(self, val_epochs=11):
        self.leopart_model.eval()
        # teacher_feature_extractor = vit_small(patch_size=16, pretrained=True)
        # msg = teacher_feature_extractor.load_state_dict(self.dense_tuning_model.get_finetuned_backbone_state_dict(), strict=False)
        # print(msg)
        # model = vit_base(patch_size=8, pretrained=False)
        # prototypes = self.dense_tuning_model.get_prototypes()
        # mlph = self.dense_tuning_model.get_mlp_head()
        # state_dict = torch.load("leopart_vitb8.ckpt")
        # model.load_state_dict({".".join(k.split(".")[1:]): v for k, v in state_dict.items()}, strict=False)
        masked_kd = mae_vit_base_patch16(loss_weights="top1", mask_type="attention", fusion_type="linear", target_norm="whiten", loss_type="smoothl1", head_type="linear")
        msg = masked_kd.load_dino_state_dict(self.leopart_model.model.state_dict(), strict=False)
        print(msg)
        mkd_model = MaskedMKD(masked_kd, 0.5)
        # mkd_model = TransformerMKD(teacher_feature_extractor, student_feature_extractor, None, None, layer_id=1)
        mkd_trainer = MKDTrainer(self.test_train_loader, self.test_dataloader, mkd_model, val_epochs, self.device, self.logger)
        optimization_config = {
            'init_lr': 1e-4,
            'peak_lr': 1e-3,
            'decay_half_life': 0,
            'warmup_steps': 0,
            'grad_norm_clip': 0,
            'init_weight_decay': 1e-2,
            'peak_weight_decay': 1e-2
        }
        mkd_trainer.setup_optimizer(optimization_config)
        mkd_trainer.train()
    

    def train(self):
        for epoch in range(self.num_epochs):
            print("Epoch: {}".format(epoch))
            if epoch % 1 == 0:
                self.validate(val_epochs=11)
            self.train_one_epoch()



def run(args):
    device = args.device
    ucf101_path = args.ucf101_path
    clip_durations = args.clip_durations
    batch_size = args.batch_size
    num_workers = args.num_workers
    input_size = args.input_size
    num_epochs = args.num_epochs
    crop_size = args.crop_size
    crop_scale = args.crop_scale_tupple
    num_ptototypes = args.num_prototypes    
    dataset = args.dataset
    config = vars(args)
    logger = wandb.init(project=project_name, group=f'{dataset}_dino_leopart_one_class', job_type='final_results', config=config)
    # transformations = transforms.Compose([
    #     transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
    #     transforms.RandomResizedCrop((input_size, input_size), scale=crop_scale),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     ## imagenet mean and std
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])
    if dataset == "fmnist" or dataset == "mnist":
        val_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        val_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    train_transform = LeopartTransforms([input_size, 96], [2, 4], [0.3, 0.1], [1., 0.3], 1, 0.01, 1)

    ################################################################
    abnormal_class = args.abnormal_class
    # normal_classes = [i for i in range(0, 10) if i == abnormal_class]
    # normal_classes = [i for i in range(1, 44)]
    # normal_classes = abnormal_class
    normal_classes = [8]
    if dataset == "fmnist":
        data_handler = FMNIST_Handler(batch_size=batch_size, normal_classes=normal_classes, transformations=train_transform, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "pascal":
        data_handler = PascalVOCDataModule(batch_size=batch_size, normal_classes=normal_classes, train_transform=train_transform, val_transform=val_transformations, test_transform=val_transformations, num_workers=num_workers)
    elif dataset == "cifar10":
        data_handler = Cifar10_Handler(batch_size=batch_size, normal_classes=normal_classes, transformations=train_transform, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "mnist":
        data_handler = MNIST_Handler(batch_size=batch_size, normal_classes=normal_classes, transformations=train_transform, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "mvtec":
        data_handler = MVTecAD_Handler(batch_size=batch_size, normal_classes=normal_classes, transformations=train_transform, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "coil":
        data_handler = Coil100_multi_object_Handler(batch_size=batch_size, train_folder="data/coil-100/train", test_folder="data/coil-100/test", transformations=train_transform, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "coco":
        data_handler = Coco_Handler(batch_size, normal_classes, train_transform, val_transformations, num_workers, device)

    train_loader = data_handler.get_train_loader()
    val_loader = data_handler.get_val_loader()
    test_loader = data_handler.get_test_loader()
    ################################################################
    
    # vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    # supervised_pretraining = vit_small_patch16_224(pretrained=True)
    # state_dict = torch.load("models/leopart_vits16.ckpt")
    # msg = vit_model.load_state_dict({".".join(k.split(".")[1:]): v for k, v in state_dict.items()}, strict=False)
    # msg = vit_model.load_state_dict(supervised_pretraining.state_dict(), strict=False)
    # print(msg)
    train_config["batch_size"] = batch_size
    train_config["nmb_prototypes"] = num_ptototypes
    train_config["max_epochs"] = num_epochs
    
    leopart = Leopart(
        use_teacher=train_config["use_teacher"],
        loss_mask=train_config["loss_mask"],
        queue_length=train_config["queue_length"],
        momentum_teacher=train_config["momentum_teacher"],
        momentum_teacher_end=train_config["momentum_teacher_end"],
        num_clusters_kmeans=train_config["num_clusters_kmeans_miou"],
        weight_decay_end=train_config["weight_decay_end"],
        roi_align_kernel_size=train_config["roi_align_kernel_size"],
        val_downsample_masks=train_config["val_downsample_masks"],
        arch=train_config["arch"],
        patch_size=train_config["patch_size"],
        lr_heads=train_config["lr_heads"],
        gpus=1,
        num_classes=20,
        batch_size=train_config["batch_size"],
        num_samples=data_handler.get_num_samples(),
        projection_feat_dim=train_config["projection_feat_dim"],
        projection_hidden_dim=train_config["projection_hidden_dim"],
        n_layers_projection_head=train_config["n_layers_projection_head"],
        max_epochs=train_config["max_epochs"],
        val_iters=train_config["val_iters"],
        nmb_prototypes=train_config["nmb_prototypes"],
        temperature=train_config["temperature"],
        sinkhorn_iterations=train_config["sinkhorn_iterations"],
        crops_for_assign=train_config["crops_for_assign"],
        nmb_crops=[2, 4],
        optimizer=train_config["optimizer"],
        exclude_norm_bias=train_config["exclude_norm_bias"],
        lr_backbone=train_config["lr_backbone"],
        final_lr=train_config["final_lr"],
        weight_decay=train_config["weight_decay"],
        epsilon=train_config["epsilon"],
    )


    w_student = get_backbone_weights(train_config["arch"],
                                        train_config["pretrained_weights"],
                                        patch_size=train_config.get("patch_size"),
                                        weight_prefix="model")
    w_teacher = get_backbone_weights(train_config["arch"],
                                        train_config["pretrained_weights"],
                                        patch_size=train_config.get("patch_size"),
                                        weight_prefix="teacher")
    leopart.load_state_dict(state_dict=[w_student, w_teacher], strict=False)

    optimization_config = {
        'init_lr': 1e-4,
        'peak_lr': 1e-3,
        'decay_half_life': 0,
        'warmup_steps': 0,
        'grad_norm_clip': 1.0,
        'init_weight_decay': 1e-2,
        'peak_weight_decay': 1e-2
    }
    
    if dataset == "fmnist" or dataset == "mnist":
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        val_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        val_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    if dataset == "fmnist":
        data_handler = FMNIST_Handler(batch_size=batch_size, normal_classes=normal_classes, transformations=transformations, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "pascal":
        data_handler = PascalVOCDataModule(batch_size=batch_size, normal_classes=normal_classes, train_transform=transformations, val_transform=val_transformations, test_transform=val_transformations, num_workers=num_workers)
    elif dataset == "cifar10":
        data_handler = Cifar10_Handler(batch_size=batch_size, normal_classes=normal_classes, transformations=transformations, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "mnist":
        data_handler = MNIST_Handler(batch_size=batch_size, normal_classes=normal_classes, transformations=transformations, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "mvtec":
        data_handler = MVTecAD_Handler(batch_size=batch_size, normal_classes=normal_classes, transformations=transformations, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "coil":
        data_handler = Coil100_multi_object_Handler(batch_size=batch_size, train_folder="data/coil-100/train", test_folder="data/coil-100/test", transformations=transformations, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "coco":
        data_handler = Coco_Handler(batch_size, normal_classes, transformations, val_transformations, num_workers, device)
    test_train_loader = data_handler.get_train_loader()
    # test_loader = data_handler.get_test_loader() ## For MVTecAD

    patch_prediction_trainer = LeopartTuningTrainer(train_loader, test_train_loader, test_loader, leopart, num_epochs, device, logger)
    patch_prediction_trainer.setup_optimizer(optimization_config)
    patch_prediction_trainer.train()

    # patch_prediction_trainer.visualize()


            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:4")
    parser.add_argument('--ucf101_path', type=str, default="Imagenet normalization")
    parser.add_argument('--clip_durations', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--crop_scale_tupple', type=tuple, default=(0.15, 1))
    parser.add_argument('--abnormal_class', type=int, default=0)
    parser.add_argument('--num_prototypes', type=int, default=20)
    parser.add_argument('--dataset', type=str, default="coco")
    args = parser.parse_args()
    run(args)