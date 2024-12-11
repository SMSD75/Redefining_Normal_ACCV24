import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from data_handler import Cifar100_Handler, Cifar10_Handler, FMNIST_Handler, MVTecAD_Handler
import torchvision.transforms as transforms
from models import FeatureExtractor
from optimizer import MKDOptimizer
from vision_transformers import vit_small, vit_base, mae_vit_base_patch16
import argparse
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch16_224, vit_small_patch16_384
import timm.optim.optim_factory as optim_factory
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image

project_name = "MKD_anomaly_detection_ivi"


class TransformerMKD(torch.nn.Module):
    def __init__(self, teacher_feature_extractor, student_feature_extractor, prototypes=None, T_MLP_head=None, layer_id=1, normalize=True) -> None:
        super(TransformerMKD, self).__init__()
        self.teacher_feature_extractor = FeatureExtractor(teacher_feature_extractor, eval_spatial_resolution=14, d_model=384)
        self.student_feature_extractor = FeatureExtractor(student_feature_extractor, eval_spatial_resolution=14, d_model=384)
        self.layer_id = layer_id ## it is fixed to 1 for now
        self.normalize = normalize
        self.teacher_feature_extractor.freeze_feature_extractor()
        self.criterion = nn.MSELoss()
        self.prototypes = prototypes
        self.T_MLP_head = T_MLP_head
        ## create a MLP head for student similar to teacher
        if self.T_MLP_head is not None:
            self.S_MLP_head =  torch.nn.Sequential(
            torch.nn.Linear(self.student_feature_extractor.d_model, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 786),
        )
        # self.S_MLP_head =  torch.nn.Sequential(
        # torch.nn.Linear(self.student_feature_extractor.d_model, 256),
        # torch.nn.GELU(),
        # torch.nn.Linear(256, 128),
        # torch.nn.GELU(),
        # torch.nn.Linear(128, 128),
        # torch.nn.GELU(),
        # torch.nn.Linear(128, 768),
        # )
    
    def forward(self, x, n=1, mask=False):
        with torch.no_grad():
            teacher_output, _ = self.teacher_feature_extractor.forward_features(x, n=n)
        if mask:
            x = self.rotate_input(x)
        student_output, _ = self.student_feature_extractor.forward_features(x, n=n)
        return student_output, teacher_output

    def freeze_teacher_prototypes_head(self):
        for param in self.T_MLP_head.parameters():
            param.requires_grad = False
        self.prototypes.requires_grad = False

    def mask_input(self, x, mask_ratio):
        bs, c, h, w = x.shape
        ## generate a random mask of size (bs, c, h, w)
        mask = torch.rand((bs, 14, 14))
        ## convert the mask to binary mask
        mask = mask < mask_ratio
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, c, 1, 1)
        resized_mask = F.interpolate(mask.float(), size=(h, w)).bool()
        ## mask the input
        x[resized_mask] = 0
        return x


    def rotate_input(self, x):
        ## generate a random degree
        degree = torch.randint(0, 4, (1,)).item()
        if degree == 2:
            degree = 0
        for j in range(degree):
            x = torch.rot90(x, 1, (2, 3))
        return x

    def train_step(self, data, target):
        self.train()
        student_output, teacher_output = self.forward(data, n=1, mask=False)
        loss = self.compute_loss(student_output, teacher_output)
        return loss

    def compute_loss(self, student_output, teacher_output):
        loss = 0
        if self.prototypes is not None:
            prj_student_output = self.S_MLP_head(student_output)
            prj_teacher_output = self.T_MLP_head(teacher_output)
            normalized_student_output = F.normalize(prj_student_output, p=2, dim=-1)
            normalized_teacher_output = F.normalize(prj_teacher_output, p=2, dim=-1)
            ## compute cosine similarity between student and teacher output
            bs, n_p, d = normalized_student_output.shape
            student_proto_sim = torch.matmul(normalized_student_output.view(-1, d), self.prototypes.T)
            teacher_proto_sim = torch.matmul(normalized_teacher_output.view(-1, d), self.prototypes.T)
            student_proto_sim = student_proto_sim.view(bs, n_p, -1)
            teacher_proto_sim = teacher_proto_sim.view(bs, n_p, -1)
            student_proto_sim = student_proto_sim.permute(0, 2, 1)
            ## computer cross entropy loss between student and teacher prototype similarity
            loss += F.cross_entropy(student_proto_sim, teacher_proto_sim.argmax(dim=-1))
        ## check if student output is a list
        if isinstance(student_output, list):
            for i in range(len(student_output)):
                if self.normalize:
                    # student_output[i] = self.S_MLP_head(student_output[i])
                    normalized_student_output = F.normalize(student_output[i], p=2, dim=-1)
                    normalized_teacher_output = F.normalize(teacher_output[i], p=2, dim=-1)
                    loss += self.criterion(normalized_student_output, normalized_teacher_output)
                else:
                    loss += self.criterion(student_output[i], teacher_output[i])
        else:
            if self.normalize:
                # student_output = self.S_MLP_head(student_output)
                normalized_student_output = F.normalize(student_output, p=2, dim=-1)
                normalized_teacher_output = F.normalize(teacher_output, p=2, dim=-1)
                loss += self.criterion(normalized_student_output, normalized_teacher_output)
            else:
                loss += self.criterion(student_output, teacher_output)
        return loss


    def get_params_dict(self, model, exclude_decay=True, lr=1e-4):
        params = []
        excluded_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if exclude_decay and (name.endswith(".bias") or (len(param.shape) == 1)):
                    excluded_params.append(param)
                else:
                    params.append(param)
                print(f"{name} is trainable")
        return [{'params': params, 'lr': lr},
                    {'params': excluded_params, 'weight_decay': 0., 'lr': lr}]

    def get_optimization_params(self):
        student_feature_extractor_params = self.get_params_dict(self.student_feature_extractor,exclude_decay=True, lr=1e-4)
        # student_mlp_head_params = self.get_params_dict(self.S_MLP_head, exclude_decay=True, lr=1e-4)
        student_mlp_head_params = []
        if self.T_MLP_head is not None:
            student_mlp_head_params = self.get_params_dict(self.S_MLP_head, exclude_decay=True, lr=1e-4)
        all_params = student_feature_extractor_params + student_mlp_head_params
        return all_params

    def val_step(self, x, target):
        self.eval()
        data_losses = 0
        student_output, teacher_output = self.forward(x, n=1)
        if self.prototypes is not None:
            student_output = self.S_MLP_head(student_output)
            teacher_output = self.T_MLP_head(teacher_output)
            normalized_student_output = F.normalize(student_output, p=2, dim=-1)
            normalized_teacher_output = F.normalize(teacher_output, p=2, dim=-1)
            ## compute cosine similarity between student and teacher output
            bs, n_p, d = normalized_student_output.shape
            student_proto_sim = torch.matmul(normalized_student_output.view(-1, d), self.prototypes.T)
            teacher_proto_sim = torch.matmul(normalized_teacher_output.view(-1, d), self.prototypes.T)
            student_proto_sim = student_proto_sim.view(bs, n_p, -1)
            teacher_proto_sim = teacher_proto_sim.view(bs, n_p, -1)
            student_proto_sim = student_proto_sim.permute(0, 2, 1)

            ## computer cross entropy loss between student and teacher prototype similarity
            loss = F.cross_entropy(student_proto_sim, teacher_proto_sim.argmax(dim=-1), reduction='none')
            data_losses = loss.mean(dim=-1).detach().cpu().numpy().tolist()
            target = target.detach().cpu().numpy().tolist()
            return data_losses, target
        

        if isinstance(student_output, list):
            for i in range(len(student_output)):
                # student_output[i] = self.S_MLP_head(student_output[i])
                student_output_i = student_output[i]
                teacher_output_i = teacher_output[i]
                if self.normalize:
                    student_output_i = F.normalize(student_output[i], p=2, dim=-1)
                    teacher_output_i = F.normalize(teacher_output[i], p=2, dim=-1)
                student_output_i = student_output_i.flatten(start_dim=1)
                teacher_output_i = teacher_output_i.flatten(start_dim=1)
                if isinstance(self.criterion, nn.MSELoss):
                    loss = F.mse_loss(student_output_i, teacher_output_i, reduction='none')
                data_losses += loss.mean(dim=-1)
            data_losses = data_losses.detach().cpu().numpy().tolist()
            target = target.detach().cpu().numpy().tolist()
        else:
            if self.normalize:
                # student_output = self.S_MLP_head(student_output)
                student_output = F.normalize(student_output, p=2, dim=-1)
                teacher_output = F.normalize(teacher_output, p=2, dim=-1)
            student_output = student_output.flatten(start_dim=1)
            teacher_output = teacher_output.flatten(start_dim=1)
            if isinstance(self.criterion, nn.MSELoss):
                loss = F.mse_loss(student_output, teacher_output, reduction='none')
            data_losses = loss.mean(dim=-1).detach().cpu().numpy().tolist()
            target = target.detach().cpu().numpy().tolist()
        return data_losses, target


class MaskedMKD(torch.nn.Module):
    def __init__(self, masked_mkd_model, mask_ratio) -> None:
        super(MaskedMKD, self).__init__()
        self.masked_mkd_model = masked_mkd_model
        self.mask_ratio = mask_ratio
    
    def train_step(self, inputs, targets):
        self.train()
        loss = self.masked_mkd_model(inputs, mask_ratio=self.mask_ratio)
        return loss
    
    def val_step(self, inputs, targets):
        self.eval()
        loss = 0
        teacher_output = self.masked_mkd_model.forward_dino_val(inputs)
        for i in range(1):
            student_output, ids_keep = self.masked_mkd_model.forward_encoder_val(inputs, mask_ratio=0)
            normalized_student_output = F.normalize(student_output, p=2, dim=-1)
            normalized_teacher_output = F.normalize(teacher_output, p=2, dim=-1)
            normalized_student_output = normalized_student_output[:, 1:, :]
            cls = normalized_teacher_output[:, :1, :]
            normalized_teacher_output = normalized_teacher_output[:, 1:, :]
            normalized_teacher_output = torch.gather(normalized_teacher_output, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, normalized_teacher_output.shape[-1]))
            # normalized_teacher_output = torch.cat([cls, normalized_teacher_output], dim=1)
            loss_i = F.mse_loss(normalized_student_output, normalized_teacher_output, reduction='none')
            loss_i = loss_i.mean([1, 2])
            loss += loss_i
        loss = loss / 1
        # loss = self.masked_mkd_model(inputs, mask_ratio=0, eval=True)
        loss = loss.detach().cpu().numpy().tolist() 
        targets = targets.detach().cpu().numpy().tolist()
        return loss, targets

    def visualize(self, inputs, label="normal"):
        self.eval()
        loss = 0
        teacher_output = self.masked_mkd_model.forward_dino_val(inputs)
        for i in range(1):
            student_output, ids_keep = self.masked_mkd_model.forward_encoder_val(inputs, mask_ratio=0)
            normalized_student_output = F.normalize(student_output, p=2, dim=-1)
            normalized_teacher_output = F.normalize(teacher_output, p=2, dim=-1)
            normalized_student_output = normalized_student_output[:, 1:, :]
            cls = normalized_teacher_output[:, :1, :]
            normalized_teacher_output = normalized_teacher_output[:, 1:, :]
            normalized_teacher_output = torch.gather(normalized_teacher_output, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, normalized_teacher_output.shape[-1]))
            # normalized_teacher_output = torch.cat([cls, normalized_teacher_output], dim=1)
            loss_i = F.mse_loss(normalized_student_output, normalized_teacher_output, reduction='none')
            loss_i = loss_i.mean([2])
            ## normalize the error values across dim=1
            loss_i = (loss_i - loss_i.min(dim=1, keepdim=True)[0]) / (loss_i.max(dim=1, keepdim=True)[0] - loss_i.min(dim=1, keepdim=True)[0])
            error_map = loss_i.reshape(-1, 14, 14)
            upsampled_error_map = F.interpolate(error_map.unsqueeze(1), size=(224, 224), mode='bilinear').squeeze(1)
            bs = upsampled_error_map.shape[0]

            for j in range(bs):
                # Ensure bool_error_map is a boolean mask
                bool_error_map = upsampled_error_map[j] > 0.6

                # Denormalize inputs
                denormalized_inputs = inputs[j].cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                
                # Ensure the denormalized inputs are in the range [0, 1] before scaling
                denormalized_inputs = denormalized_inputs.clamp(0, 1)

                # Scale to [0, 255] and convert to uint8
                denormalized_inputs_uint8 = (denormalized_inputs * 255).to(torch.uint8)

                # Convert the uint8 tensor to PIL Image for wandb
                denormalized_inputs_pil = to_pil_image(denormalized_inputs_uint8)

                # Log the original image
                wandb.log({f"{label}_image": wandb.Image(denormalized_inputs_pil)})

                # Log the error map
                error_map_pil = to_pil_image(upsampled_error_map[j].cpu())
                wandb.log({f"{label}_error_map": wandb.Image(error_map_pil)})
                
                # Create the overlay image, using the uint8 image tensor
                overlay_img = draw_segmentation_masks(denormalized_inputs_uint8, bool_error_map.cpu(), alpha=0.5, colors=["red", "green"])

                # Convert the overlay tensor to PIL Image for wandb
                overlay_pil = to_pil_image(overlay_img)
                wandb.log({f"{label}_overlay": wandb.Image(overlay_pil)})

    def get_optimizer(self):
        param_groups = optim_factory.param_groups_weight_decay(self.masked_mkd_model, 0.05, no_weight_decay_list=["distill_weights"])
        optimizer = torch.optim.AdamW(param_groups, lr=1e-5, betas=(0.9, 0.95))
        return optimizer

class MKDTrainer:
     
    def __init__(self, train_loader, test_dataloader, mkd_model, num_epochs, device, logger):
        self.train_loader = train_loader
        self.test_dataloader = test_dataloader
        self.mkd_model = mkd_model
        self.device = device
        self.mkd_model = self.mkd_model.to(self.device)
        self.optimizer = None
        self.num_epochs = num_epochs
        self.logger = logger
        self.logger.watch(mkd_model, log="all", log_freq=10)  
    
    def setup_optimizer(self, optimization_config):
        # model_params = self.mkd_model.get_optimization_params()
        # init_lr = optimization_config['init_lr']
        # peak_lr = optimization_config['peak_lr']
        # decay_half_life = optimization_config['decay_half_life']
        # warmup_steps = optimization_config['warmup_steps']
        # grad_norm_clip = optimization_config['grad_norm_clip']
        # init_weight_decay = optimization_config['init_weight_decay']
        # peak_weight_decay = optimization_config['peak_weight_decay']
        # ## read the first batch from dataloader to get the number of iterations
        # num_itr = len(self.train_loader) + 1
        # max_itr = self.num_epochs * num_itr
        # self.optimizer = MKDOptimizer(model_params, init_lr, peak_lr, warmup_steps, grad_norm_clip, max_itr)
        # self.optimizer.setup_optimizer()
        # self.optimizer.setup_scheduler()
        self.optimizer = self.mkd_model.get_optimizer()
    

    def train_one_epoch(self, epoch):
        self.mkd_model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            loss = self.mkd_model.train_step(data, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.optimizer.update_lr()
            epoch_loss += loss.item()
            # lr = self.optimizer.get_lr()
            # self.logger.log({"lr": lr})
            if batch_idx % 2 == 0:
                # self.logger.log({"train_loss": loss})
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss))
        epoch_loss /= len(self.train_loader)
        print("Epoch Loss: {}".format(epoch_loss))
    
    
    def train(self):
        for epoch in range(self.num_epochs):
            if epoch == 0 or epoch  == self.num_epochs - 1:
                self.validate(epoch)
            print("Epoch: {}".format(epoch))
            self.train_one_epoch(epoch)
    
    def validate(self, epoch):
        self.mkd_model.eval()
        test_loss = 0
        correct = 0
        dataset_losses = []
        dataset_targets = []
        normal_inputs = []
        abnormal_inputs = []
        with torch.no_grad():
            for (data, target) in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                target.squeeze_()
                if len(normal_inputs) <= 3:
                    if torch.any(target == 0):
                        normal_inputs.append(data[target == 0])
                if len(abnormal_inputs) == 0:
                    if torch.any(target == 1):
                        abnormal_inputs.append(data[target == 1])
                data_losses, target = self.mkd_model.val_step(data, target)
                dataset_losses += data_losses
                dataset_targets+= target 
        dataset_targets = np.array(dataset_targets)
        dataset_losses = np.array(dataset_losses)
        fpr, tpr, thresholds = metrics.roc_curve(dataset_targets, dataset_losses, pos_label=1)
        self.logger.log({"roc_auc_score": metrics.auc(fpr, tpr)})
        self.mkd_model.visualize(normal_inputs[0], "normal")
        self.mkd_model.visualize(normal_inputs[1], "normal")
        self.mkd_model.visualize(normal_inputs[2], "normal")
        self.mkd_model.visualize(abnormal_inputs[0], "abnormal")
        print(f"Epoch : {epoch}, ROC AUC Score: ", metrics.auc(fpr, tpr))

    

def main(params_dict):
    num_epochs = params_dict["num_epochs"]
    batch_size = params_dict["batch_size"]
    num_workers = params_dict["num_workers"]
    layer_id = params_dict["layer_id"]
    logging_dir = params_dict["logging_directory"]
    logging_dir = os.path.join(logging_dir, datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    pretraining = params_dict["pretraining"]
    pretraining_path = params_dict["checkpoint"]
    criterion = params_dict["criterion"]
    abnormal_class = params_dict["abnormal_class"]
    loss_weights = params_dict["loss_weights"]
    mask_type = params_dict["mask_type"]
    fusion_type = params_dict["fusion_type"]
    target_norm = params_dict["target_norm"]
    loss_type = params_dict["loss_type"]
    head_type = params_dict["head_type"]
    mask_ratio = params_dict["mask_ratio"]
    img_size = params_dict["img_size"]
    config = vars(args)
    ## make a string of today's date
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    logger = wandb.init(project=project_name, group=d1, job_type='hyper-parameter_experiments_cifar10', config=config)
    normal_classes = [i for i in range(10) if i == abnormal_class]
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        # lambda x: x.repeat(3, 1, 1),
        ## imagenet mean and std
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        # lambda x: x.repeat(3, 1, 1),
        ## imagenet mean and std
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_handler = Cifar10_Handler(batch_size, normal_classes, transformations, val_transformations, num_workers, device)
    train_loader = data_handler.get_train_loader()
    val_loader = data_handler.get_val_loader()
    test_loader = data_handler.get_test_loader()
    masked_KD = mae_vit_base_patch16(loss_weights=loss_weights, mask_type=mask_type, fusion_type=fusion_type, target_norm=target_norm, loss_type=loss_type,
                 head_type=head_type)
    # teacher_feature_extractor = vit_small_patch16_224(pretrained=True)
    teacher_feature_extractor = vit_small(pretrained=True)
    # msg = teacher_feature_extractor.load_state_dict(pretraining.state_dict(), strict=False)
    # print(msg)
    student_feature_extractor = vit_small_patch16_224(pretrained=False)
    mkd_model = TransformerMKD(teacher_feature_extractor, student_feature_extractor, layer_id=layer_id)
    # masked_mkd_model = MaskedMKD(masked_KD, mask_ratio)
    mkd_trainer = MKDTrainer(train_loader, test_loader, mkd_model, num_epochs, device, logger)
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Anomaly Detection')
    parser.add_argument('--logging_directory', type=str, default='logs')
    parser.add_argument('--num_epochs', type=int, default=11,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loading (default: 4)')
    parser.add_argument('--layer_id', type=int, default=1, help='layer id to extract features from (default: 1)')
    parser.add_argument('--pretraining', type=str, default="leopart", help='pre-training method (default: dino)')
    parser.add_argument('--distillation_loss', type=str, default="MSE", help='distillation loss (default: MSE)')
    parser.add_argument('--checkpoint', type=str, default="Temp/model_3.44801375967391.pth", help='path to checkpoint')
    parser.add_argument('--criterion', type=str, default="normalized_mse", help='training loss criteria')
    parser.add_argument('--abnormal_class', type=int, default=9, help='abnormal class')
    parser.add_argument('--loss_weights', type=str, default="top5", help='loss weights')
    parser.add_argument('--mask_type', type=str, default="attention", help='mask type')
    parser.add_argument('--fusion_type', type=str, default="linear", help='fusion type')
    parser.add_argument('--target_norm', type=str, default="whiten", help='target norm')
    parser.add_argument('--loss_type', type=str, default="smoothl1", help='loss type')
    parser.add_argument('--head_type', type=str, default="linear", help='head type')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')
    parser.add_argument('--img_size', type=int, default=384, help='image size')
    args = parser.parse_args()
    params = vars(args)
    for i in range(0, 10):
        params['abnormal_class'] = i
        main(params)

    
    

    