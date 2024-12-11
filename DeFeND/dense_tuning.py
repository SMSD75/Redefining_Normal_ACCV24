from matplotlib import patches, pyplot as plt
import torch
import argparse
import os
from sqlite3 import Time
import time
import torch
from torchvision import transforms
import torch.nn.functional as F
from data_handler import Cifar100_Handler, Cifar10_Handler, FMNIST_Handler, MVTecAD_Handler
from main import MKDTrainer, TransformerMKD, MaskedMKD
from models import CrossAttentionBlock, FeatureExtractor
import wandb
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from optimizer import PatchCorrespondenceOptimizer
import torchvision.transforms as trn
from torch import distributed as dist
from data import PascalVOCDataModule

from transformations import DenseTransforms
from transformers import ViTModel
from timm.models.vision_transformer import vit_small_patch16_224, vit_small_patch8_224, vit_base_patch16_384

from vision_transformers import vit_small, mae_vit_base_patch16
import numpy as np



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


@torch.no_grad()
def sinkhorn(Q: torch.Tensor, nmb_iters: int, world_size=1) -> torch.Tensor:
    with torch.no_grad():
        Q = Q.detach().clone()
        sum_Q = torch.sum(Q)
        if world_size > 1:
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        K, B = Q.shape
        u = torch.zeros(K).to(Q.device)
        r = torch.ones(K).to(Q.device) / K
        c = torch.ones(B).to(Q.device) / B * world_size

        if world_size > 1:
            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

        for _ in range(nmb_iters):
            if world_size > 1:
                u = curr_sum
            else:
                u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            if world_size > 1:
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def find_optimal_assignment(scores, epsilon, sinkhorn_iterations):
    """
    Computes the Sinkhorn matrix Q.
    :param scores: similarity matrix
    :return: Sinkhorn matrix Q
    """
    with torch.no_grad():
        q = torch.exp(scores / epsilon).t()
        q = sinkhorn(q, sinkhorn_iterations, world_size=1)
        # q = torch.softmax(scores / epsilon, dim=0)
        # q = q / q.sum(dim=1, keepdim=True)
    return q
    
def generate_random_crop(img, crop_size):
    ## generate a random crop mask with all 1s on the img with  crop_scale=(0.05, 0.3), and size crop_size
    bs, c, h, w = img.shape
    crop = torch.zeros((bs, h, w))
    x = torch.randint(0, h - crop_size, (1,)).item()
    y = torch.randint(0, w - crop_size, (1,)).item()
    crop[:, x:x + crop_size, y:y + crop_size] = 1
    return crop


def random_crop_mask(img, aspect_ratio_range=(3/4, 4/3), scale_range=(0.05, 0.3), mask_height=None, mask_width=None):
    """
    Generate a random crop mask with a given aspect ratio.
    
    H, W: Image dimensions
    aspect_ratio: Desired aspect ratio = mask_width/mask_height
    mask_height or mask_width: Specify one of them and the other will be computed using aspect_ratio.
    """
    bs, c, H, W = img.shape
    # Extract values from the provided ranges
    min_aspect, max_aspect = aspect_ratio_range
    min_scale, max_scale = scale_range
    
    # Randomly select an aspect ratio and scale within the provided range
    random_aspect_ratio = torch.rand(1).item() * (max_aspect - min_aspect) + min_aspect
    random_scale = torch.rand(1).item() * (max_scale - min_scale) + min_scale
    
    # Compute mask dimensions based on random scale and aspect ratio
    mask_width = int(W * random_scale * random_aspect_ratio)
    mask_height = int(W * random_scale / random_aspect_ratio)

    # Check if mask dimensions exceed the image dimensions
    if mask_height > H or mask_width > W:
        raise ValueError("Mask dimensions exceed the image dimensions.")

    # Generate a random starting point
    y = torch.randint(0, H - mask_height + 1, (1,)).item()
    x = torch.randint(0, W - mask_width + 1, (1,)).item()

    # Create the mask with all zeros and set the crop area to ones
    mask = torch.zeros((bs, H, W))
    mask[:, y:y+mask_height, x:x+mask_width] = 1

    return mask

## a function that generates random crop masks for a batch of images
def generate_random_crop_masks(imgs, aspect_ratio_range=(3/4, 4/3), scale_range=(0.05, 0.3), mask_height=None, mask_width=None):

    bs, c, H, W = imgs.shape
    # Extract values from the provided ranges
    min_aspect, max_aspect = aspect_ratio_range
    min_scale, max_scale = scale_range

    # Randomly select an aspect ratio and scale within the provided range
    random_aspect_ratio = torch.rand(1).item() * (max_aspect - min_aspect) + min_aspect
    random_scale = torch.rand(1).item() * (max_scale - min_scale) + min_scale

    # Compute mask dimensions based on random scale and aspect ratio
    mask_width = int(W * random_scale * random_aspect_ratio)
    mask_height = int(W * random_scale / random_aspect_ratio)

    # Check if mask dimensions exceed the image dimensions
    if (mask_height > H) or (mask_width > W):
        raise ValueError("Mask dimensions exceed the image dimensions.")
    

    y_uniform = torch.rand(bs).to(imgs.device)
    x_uniform = torch.rand(bs).to(imgs.device)
    
    # Scale and shift to the desired range [0, H - mask_height[i] + 1) for each i
    y = (y_uniform * (H - mask_height + 1)).floor().long()
    x = (x_uniform * (W - mask_width + 1)).floor().long()

    x = (x // 16) * 16
    y = (y // 16) * 16

    # Create the mask with all zeros and set the crop area to ones
    mask = torch.zeros((bs, H, W)).to(imgs.device)
    for i in range(bs):
        mask[i, y[i]:y[i]+mask_height, x[i]:x[i]+mask_width] = 1

    return mask



class CorrespondenceDetection():
    def __init__(self, window_szie, spatial_resolution=14, output_resolution=96) -> None:
        self.window_size = window_szie
        self.neihbourhood = self.restrict_neighborhood(spatial_resolution, spatial_resolution, self.window_size)
        self.output_resolution = output_resolution

    
    def restrict_neighborhood(self, h, w, size_mask_neighborhood):
        # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
        mask = torch.zeros(h, w, h, w)
        for i in range(h):
            for j in range(w):
                for p in range(2 * size_mask_neighborhood + 1):
                    for q in range(2 * size_mask_neighborhood + 1):
                        if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                            continue
                        if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                            continue
                        mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

        # mask = mask.reshape(h * w, h * w)
        return mask

    def __call__(self, features1, features2, crops):
        bs, spatial_resolution, spatial_resolution, d_model = features1.shape
        _, h, w = crops.shape
        patch_size = h // spatial_resolution
        crops = crops.reshape(bs, h // patch_size, patch_size, w // patch_size, patch_size).permute(0, 1, 3, 2, 4)
        crops = crops.flatten(3, 4)
        cropped_feature_mask = crops.sum(-1) > 0 ## size (bs, spatial_resolution, spatial_resolution)
        ## find the idx of the croped features_mask
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        similarities = torch.einsum('bxyd,bkzd->bxykz', features1, features2)
        most_similar_features_mask = torch.zeros(bs, spatial_resolution, spatial_resolution)
        revised_crop = torch.zeros(bs, spatial_resolution, spatial_resolution)
        self.neihbourhood = self.neihbourhood.to(features1.device)
        similarities = similarities * self.neihbourhood.unsqueeze(0)
        similarities = similarities.flatten(3, 4)
        most_similar_cropped_patches_list = []
        # for i, crp_feature_mask in enumerate(cropped_feature_mask): 
        crp_feature_mask = cropped_feature_mask[0]
        true_coords  = torch.argwhere(crp_feature_mask)
        min_coords = true_coords.min(0).values
        max_coords = true_coords.max(0).values
        rectangle_shape = max_coords - min_coords + 1
        crop_h, crop_w = rectangle_shape
        most_similar_patches = similarities.argmax(-1)
        most_similar_cropped_patches = most_similar_patches[cropped_feature_mask]
        most_similar_cropped_patches = most_similar_cropped_patches.reshape(bs, crop_h, crop_w)
        # most_similar_cropped_patches = F.interpolate(most_similar_cropped_patches.float().unsqueeze(0).unsqueeze(0), size=(self.output_resolution, self.output_resolution), mode='nearest').squeeze(0).squeeze(0)
        # most_similar_cropped_patches_list.append(most_similar_cropped_patches)

        # for i, similarity in enumerate(similarities):
        #     croped_feature_idx = croped_feature_mask[i].nonzero()
        #     for j, mask_idx in enumerate(croped_feature_idx):
        #         # print(mask_idx)
        #         revised_crop[i, mask_idx[0], mask_idx[1]] = 1
        #         min_x, max_x = max(0, mask_idx[0] - self.window_size), min(spatial_resolution, mask_idx[0] + self.window_size)
        #         min_y, max_y = max(0, mask_idx[1] - self.window_size), min(spatial_resolution, mask_idx[1] + self.window_size)
        #         neiborhood_similarity = similarity[mask_idx[0], mask_idx[1], min_x:max_x, min_y:max_y]
        #         max_value = neiborhood_similarity.max()
        #         indices = (neiborhood_similarity == max_value).nonzero()[0]
        #         label_patch_number = (indices[0] + min_x) * spatial_resolution + (indices[1] + min_y)
        #         most_similar_features_mask[i, mask_idx[0], mask_idx[1]] = label_patch_number
        
        # most_similar_cropped_patches = torch.stack(most_similar_cropped_patches_list)
        revised_crop = cropped_feature_mask.float() 
        return most_similar_cropped_patches, revised_crop
    



class DenseTuning(torch.nn.Module):
    def __init__(self, input_size, vit_model, num_prototypes=20, prediction_window_size=4, logger=None):
        super(DenseTuning, self).__init__()
        self.input_size = input_size
        self.eval_spatial_resolution = input_size // 16
        self.feature_extractor = FeatureExtractor(vit_model, eval_spatial_resolution=self.eval_spatial_resolution, d_model=768)
        self.prediction_window_size = prediction_window_size
        self.CorDet = CorrespondenceDetection(window_szie=self.prediction_window_size)
        self.logger = logger
        self.num_prototypes = num_prototypes
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_extractor.d_model, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 256),
        )
        self.feature_extractor.freeze_feature_extractor(["blocks.11", "blocks.10"])
        self.prototypes = torch.nn.Parameter(torch.randn(num_prototypes, 256))
    

    def normalize_prototypes(self):
        with torch.no_grad():
            w = self.prototypes.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.copy_(w)
            
    def forward_backbone(self, imgs1, n=1):
        bs, c, h, w = imgs1.shape
        img1_features, img1_attention = self.feature_extractor.forward_features(imgs1, n=n)
        img1_features = img1_features.reshape(bs, self.eval_spatial_resolution, self.eval_spatial_resolution, self.feature_extractor.d_model)
        return img1_features
    
    def forward_sailiancy(self, img1, img2, crops):
        img1_features, img2_features = self.forward_backbone(img1, img2)
        sailiancy, revised_crop = self.CorDet(img1_features, img2_features, crops)
        return sailiancy, revised_crop


    def get_prototypes(self):
        ## detach the prototypes from the graph
        return self.prototypes.detach().clone()

    def get_mlp_head(self):
        mlp_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_extractor.d_model, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 256),
        )

        ## load the weights of the mlp head
        mlp_head.load_state_dict(self.mlp_head.state_dict())
        return mlp_head

    
    def mask_features(self, features, percentage=0.8):
        """
        features: [bs, np, d_model]
        """
        bs, np, d_model = features.shape
        ## select 0.2 of the features randomly for each sample
        mask = torch.zeros(bs, np).to(features.device)
        ids = torch.randperm(np)[:int(np * percentage)]
        mask[:, ids] = 1
        mask = mask.unsqueeze(-1).repeat(1, 1, d_model)
        features = features * mask
        return features
    
    def cross_attention(self, query, key, value):
        """
        query: [bs, nq, d_model]
        key: [bs, np, d_model]
        value: [bs, np, d_model]

        return: [bs, nq, d_model]
        """
        # Parameters
        output = self.cross_attention_layer(query, key, value)
        return output


    def visualized_crop_bbox_imag(self, crop, bin_seg_map, img):
        # Convert img to range [0, 255]
        img = img * 255
        img = img.permute(1, 2, 0)
        img_cpu = img.cpu().numpy().astype('uint8')

        # Convert bin_seg_map to range [0, 255]
        bin_seg_map = bin_seg_map * 255
        bin_seg_map = bin_seg_map.permute(1, 2, 0)
        bin_seg_map_cpu = bin_seg_map.cpu().numpy().astype('uint8')

        # Convert crop to range [0, 255]
        crop_cpu = crop * 255
        crop_cpu = crop_cpu.permute(1, 2, 0)
        crop_cpu = crop_cpu.cpu().numpy().astype('uint8')

        # Plot the figures
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(img_cpu)
        ax[1].imshow(bin_seg_map_cpu.squeeze(), cmap='gray')  # Squeeze to remove the single channel and use a gray colormap
        ax[2].imshow(crop_cpu)

        plt.show()
        fig.savefig("crop_bbox.png")
        plt.close(fig)
    
    def forward(self, imgs1, crop_list, bboxs):  
        self.normalize_prototypes()
        bs = imgs1.shape[0]
        # crop = random_crop_mask(imgs1)
        img1_features = self.forward_backbone(imgs1, n=1)
        patch_size = imgs1.size(-1) // self.feature_extractor.eval_spatial_resolution
        cropped_labels_list = []


        pred_loss = 0
        clustering_loss = 0

        for i in range(len(crop_list)):
            crops = crop_list[i]
            bbox = bboxs[:, i].long()
            # bb = bbox[0]
            # crop = torch.zeros((1, imgs1.size(-2), imgs1.size(-1)))
            # crop[:, bb[1]:bb[3], bb[0]:bb[2]] = 1
            # self.visualized_crop_bbox_imag(crops[0], crop, imgs1[0])
            cropped_labels_list = []
            for j, bb in enumerate(bbox):
                crop = torch.zeros((1, imgs1.size(-2), imgs1.size(-1)))
                crop[:, bb[1]:bb[3], bb[0]:bb[2]] = 1
                sailiancy_h = bb[3] - bb[1]
                sailiancy_w = bb[2] - bb[0]
                sailiancy = crop.reshape(1 , imgs1.size(-2) // patch_size, patch_size, imgs1.size(-1) // patch_size, patch_size).permute(0, 1, 3, 2, 4)
                sailiancy = sailiancy.flatten(3, 4)
                sailiancy = sailiancy.sum(-1) > 0 ## size (bs, spatial_resolution, spatial_resolution)
                sailiancy = sailiancy.flatten()
                sailiancy = sailiancy.nonzero(as_tuple=True)[0]
                sailiancy_h = sailiancy_h // patch_size  
                sailiancy_w = sailiancy_w // patch_size
                sailiancy = sailiancy.reshape(1, sailiancy_h, sailiancy_w)
                cropped_labels = sailiancy
                cropped_labels = torch.nn.functional.interpolate(cropped_labels.float().unsqueeze(1), size=(96, 96), mode='nearest').squeeze(1)
                # cropped_labels = torch.arange(0, 196).reshape(14, 14).unsqueeze(0).repeat(bs, 1, 1)
                cropped_labels = cropped_labels.long().to(imgs1.device)
                cropped_labels_list.append(cropped_labels)
            cropped_labels = torch.stack(cropped_labels_list).squeeze(1)
            cropped_area = crops
            cropped_area_features, _ = self.feature_extractor.forward_features(cropped_area, n=1) ## size (bs, 36, d_model)


            cropped_area_features = self.mlp_head(cropped_area_features)
            cropped_area_features = cropped_area_features.reshape(bs, 6, 6, -1).permute(0, 3, 1, 2)
            cropped_area_features = F.interpolate(cropped_area_features, size=(96, 96), mode='bilinear').permute(0, 2, 3, 1)
            cropped_area_features = cropped_area_features.flatten(0, 2)
            normalized_crop_features = F.normalize(cropped_area_features, dim=-1)
            crop_scores = torch.einsum('bd,nd->bn', normalized_crop_features , self.prototypes)
            crop_scores = crop_scores.reshape(bs, 96, 96, self.num_prototypes).permute(0, 3, 1, 2)
            ## replace numbers in cropped_labels with the corresponding prototype idx in q

            projected_img1_features = self.mlp_head(img1_features)
            projected_img1_features = F.normalize(projected_img1_features, dim=-1)
            scores = torch.einsum('bd,nd->bn', projected_img1_features.flatten(0, -2), self.prototypes)
            q = find_optimal_assignment(scores, 0.05, 3)
            q = q.reshape(bs, self.eval_spatial_resolution, self.eval_spatial_resolution, self.num_prototypes).permute(0, 3, 1, 2)
            q = q.argmax(1)
            cropped_q_gt = q.flatten(1, 2)[torch.arange(q.size(0)).unsqueeze(1), sailiancy.flatten(1, 2)].reshape(bs, sailiancy.size(-2), sailiancy.size(-1))
            resized_crop_gt = F.interpolate(cropped_q_gt.float().unsqueeze(1), size=(96, 96), mode='nearest').squeeze(1)
            clustering_loss += self.criterion(crop_scores / 0.1, resized_crop_gt.long())
        return clustering_loss / len(crop_list)
        
    def train_step(self, imgs1 , crop_list, bboxs):
        clustering_loss = self.forward(imgs1, crop_list, bboxs)
        return clustering_loss
    

    def get_optimization_params(self):
        ## print feature extractor trainable parameters
        for name, param in self.feature_extractor.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
        return [
            {"params": self.feature_extractor.parameters(), "lr": 1e-5},
            {"params": self.prototypes, "lr": 1e-4},
            {"params": self.mlp_head.parameters(), "lr": 1e-4},
        ]


    def validate_step(self, img):
        self.feature_extractor.eval()
        with torch.no_grad():
            spatial_features, _ = self.feature_extractor.forward_features(img)  # (B, np, dim)
        return spatial_features
    

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def get_finetuned_backbone_state_dict(self):
        return self.feature_extractor.model.state_dict()


        

class DenseTuningTrainer():
    def __init__(self, dataloader, test_train_loader, test_dataloader, dense_tuning_model, num_epochs, device, logger):
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.dense_tuning_model = dense_tuning_model
        self.device = device
        self.dense_tuning_model = self.dense_tuning_model.to(self.device)
        self.test_train_loader = test_train_loader
        self.optimizer = None
        self.num_epochs = num_epochs
        self.logger = logger
        # self.logger.watch(dense_tuning_model, log="all", log_freq=10)
    
    def visualize(self):
        for i, batch in enumerate(self.dataloader):
            datum, annotations = batch
            annotations = annotations.squeeze(1)
            datum = datum.squeeze(1)
            imgs1, imgs2 = datum[:, 0], datum[:, 1]
            imgs1, imgs2 = imgs1.to(self.device), imgs2.to(self.device)
            crop = random_crop_mask(imgs1)
            self.dense_tuning_model.visualize(imgs1, imgs2, crop, i)
    
    def setup_optimizer(self, optimization_config):
        model_params = self.dense_tuning_model.get_optimization_params()
        init_lr = optimization_config['init_lr']
        peak_lr = optimization_config['peak_lr']
        decay_half_life = optimization_config['decay_half_life']
        warmup_steps = optimization_config['warmup_steps']
        grad_norm_clip = optimization_config['grad_norm_clip']
        init_weight_decay = optimization_config['init_weight_decay']
        peak_weight_decay = optimization_config['peak_weight_decay']
        ## read the first batch from dataloader to get the number of iterations
        num_itr = len(self.dataloader)
        max_itr = self.num_epochs * num_itr
        ## print model parameters
        self.optimizer = PatchCorrespondenceOptimizer(model_params, init_lr, peak_lr, decay_half_life, warmup_steps, grad_norm_clip, init_weight_decay, peak_weight_decay, max_itr)
        self.optimizer.setup_optimizer()
        self.optimizer.setup_scheduler()
    

    def train_one_epoch(self):
        self.dense_tuning_model.train()
        epoch_loss = 0
        before_loading_time = time.time()
        for i, batch in enumerate(self.dataloader):
            after_loading_time = time.time()
            print("Loading Time: {}".format(after_loading_time - before_loading_time))
            data, annotations = batch
            batch_crop_list, label = data
            global_crops_1 = batch_crop_list[0]
            imgs1 = global_crops_1
            # imgs1, imgs2 = datum, datum
            imgs1 = imgs1.to(self.device)
            for j, crop in enumerate(batch_crop_list):
                batch_crop_list[j] = crop.to(self.device)
            clustering_loss = self.dense_tuning_model.train_step(imgs1, batch_crop_list[1:], label["bbox"][:, 1:, ])
            total_loss = clustering_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.update_lr()
            epoch_loss += total_loss.item()
            print("Iteration: {} Loss: {}".format(i, total_loss.item()))
            self.logger.log({"clustering_loss": clustering_loss.item()})
            lr = self.optimizer.get_lr()
            self.logger.log({"lr": lr})
            before_loading_time = time.time()
        epoch_loss /= (i + 1)
        print("Epoch Loss: {}".format(epoch_loss))


    def validate(self, val_epochs=11):
        self.dense_tuning_model.eval()
        # teacher_feature_extractor = vit_small(patch_size=16, pretrained=True)
        # msg = teacher_feature_extractor.load_state_dict(self.dense_tuning_model.get_finetuned_backbone_state_dict(), strict=False)
        # print(msg)
        # student_feature_extractor = vit_small(patch_size=16, pretrained=False)
        # prototypes = self.dense_tuning_model.get_prototypes()
        # mlph = self.dense_tuning_model.get_mlp_head()
        masked_kd = mae_vit_base_patch16(loss_weights="top1", mask_type="attention", fusion_type="linear", target_norm="whiten", loss_type="smoothl1", head_type="linear")
        msg = masked_kd.load_dino_state_dict(self.dense_tuning_model.get_finetuned_backbone_state_dict(), strict=False)
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
                self.validate()
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
    logger = wandb.init(project=project_name, group=f'{dataset}_vitb16-224_multi_class_bs=64', job_type='final_results', config=config)
    # transformations = transforms.Compose([
    #     transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
    #     transforms.RandomResizedCrop((input_size, input_size), scale=crop_scale),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     ## imagenet mean and std
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])
    if dataset == "fmnist":
        val_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        val_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    train_transform = DenseTransforms([input_size, 96], [1, 4], [0.25, 0.1], [1., 0.3], 1, 0.01, 1)

    ################################################################
    abnormal_class = args.abnormal_class
    normal_classes = [i for i in range(20) if i != abnormal_class]
    # normal_classes = [0, 1, 2, 3, 4, 5]
    if dataset == "fmnist":
        data_handler = FMNIST_Handler(batch_size=batch_size, normal_classes=normal_classes, transformations=train_transform, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "pascal":
        data_handler = PascalVOCDataModule(batch_size=batch_size, normal_classes=normal_classes, train_transform=train_transform, val_transform=val_transformations, test_transform=val_transformations, num_workers=num_workers)
    train_loader = data_handler.get_train_loader()
    val_loader = data_handler.get_val_loader()
    test_loader = data_handler.get_test_loader()
    ################################################################
    
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    # supervised_pretraining = vit_small_patch16_224(pretrained=True)
    # state_dict = torch.load("models/leopart_vits16.ckpt")
    # msg = vit_model.load_state_dict({".".join(k.split(".")[1:]): v for k, v in state_dict.items()}, strict=False)
    # msg = vit_model.load_state_dict(supervised_pretraining.state_dict(), strict=False)
    # print(msg)
    patch_prediction_model = DenseTuning(input_size, vit_model, logger=logger, num_prototypes=num_ptototypes)
    optimization_config = {
        'init_lr': 1e-4,
        'peak_lr': 1e-3,
        'decay_half_life': 0,
        'warmup_steps': 0,
        'grad_norm_clip': 1.0,
        'init_weight_decay': 1e-2,
        'peak_weight_decay': 1e-2
    }
    
    if dataset == "fmnist":
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        val_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        val_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ## imagenet mean and std
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if dataset == "fmnist":
        data_handler = FMNIST_Handler(batch_size=batch_size, normal_classes=normal_classes, transformations=transformations, val_transformations=val_transformations, num_workers=num_workers)
    elif dataset == "pascal":
        data_handler = PascalVOCDataModule(batch_size=batch_size, normal_classes=normal_classes, train_transform=transformations, val_transform=val_transformations, test_transform=val_transformations, num_workers=num_workers)
    test_train_loader = data_handler.get_train_loader()
    # test_loader = data_handler.get_test_loader() ## For MVTecAD

    patch_prediction_trainer = DenseTuningTrainer(train_loader, test_train_loader, test_loader, patch_prediction_model, num_epochs, device, logger)
    patch_prediction_trainer.setup_optimizer(optimization_config)
    patch_prediction_trainer.train()

    # patch_prediction_trainer.visualize()


            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--ucf101_path', type=str, default=" i % 2 == 1")
    parser.add_argument('--clip_durations', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--crop_scale_tupple', type=tuple, default=(0.15, 1))
    parser.add_argument('--abnormal_class', type=int, default=0)
    parser.add_argument('--num_prototypes', type=int, default=20)
    parser.add_argument('--dataset', type=str, default="pascal")
    args = parser.parse_args()
    run(args)



        


