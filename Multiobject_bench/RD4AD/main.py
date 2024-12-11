import torch
from dataset import *
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader,Dataset
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2, BN_layer, AttnBottleneck
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test, custom_eval
from torch.nn import functional as F
from torchvision import transforms


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def train(dataset_name, _class_):

    print(f" Normal : {_class_}")
    epochs = 1
    learning_rate = 0.005
    batch_size = 32
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running {dataset_name} on {device}")

    train_dataloader, test_dataloader = load_data(dataset_name = dataset_name, normal_class = _class_, batch_size = batch_size)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img, _ in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        if (epoch + 1) % 10 == 0:
            if dataset_name == 'mvtec':
                auc = evaluation(encoder, bn, decoder, test_dataloader, device)
            else:
                auc = custom_eval(encoder, bn, decoder, test_dataloader, device)
            print('AUC: {:.4f}'.format(auc))

    return encoder, bn, decoder
  

if __name__ == "__main__":

    test_aucs = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seed(0)

    dataset_name = 'pvoc'

    if dataset_name == 'cifar10' or dataset_name == 'mnist' or dataset_name == 'fashionmnist':
        classes = range(10)
    elif dataset_name == 'cifar100' or dataset_name == 'pvoc':
        classes = range(20)
    elif dataset_name == 'mvtec':
        classes = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill', 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']

    ### Unimodal Setting ###

    for _class_ in classes:
        encoder, bn, decoder = train(dataset_name, _class_)
        if dataset_name == 'mvtec':
            test_auc = test(_class_)
        else:
            test_auc = custom_eval(encoder, bn, decoder, dataset_name, _class_, device)
        test_aucs.append(test_auc)


    ### Multimodal Setting ### 

    # for abnormal_class in classes:
    #     _normal_classes = list(classes)
    #     _normal_classes.pop(abnormal_class)
    #     _class_ = _normal_classes
    #     encoder, bn, decoder = train(dataset_name, _class_)
    #     if dataset_name == 'mvtec':
    #         test_auc = test(_class_)
    #     else:
    #         test_auc = custom_eval(encoder, bn, decoder, dataset_name, _class_, device)
    #     test_aucs.append(test_auc)

    print(f"Test AUCs {test_aucs}")
    print(f"Mean AUC {np.mean(test_aucs)}")
