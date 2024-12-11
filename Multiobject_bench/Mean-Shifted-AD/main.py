import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import utils
from tqdm import tqdm
import torch.nn.functional as F


def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

def train_model(model, train_loader, test_loader, train_loader_1, device, args):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    if args.angular:
        center = F.normalize(center, dim=-1)
    center = center.to(device)
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader_1, optimizer, center, device, args.angular)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
    auc, _ = get_score(model, device, train_loader, test_loader)
    print(f"AUC after {args.epochs} epochs:  {auc}")
    #print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
    return auc

def run_epoch(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0

    for ((img1, img2), _) in tqdm(train_loader, desc='Train...'):
        img1, img2 = img1.to(device), img2.to(device)
        optimizer.zero_grad()
        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center
        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()
        optimizer.step()
        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)

def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.Model(args.backbone)
    model = model.to(device)

    train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone)
    auc = train_model(model, train_loader, test_loader, train_loader_1, device, args)
    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10', help='Choose dataset')
    parser.add_argument('--epochs', default=10, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--backbone', default='vit-base', type=str, help='Model to use')
    parser.add_argument('--angular', action='store_true', help='Train with angular center loss')
    args = parser.parse_args()
    
    if args.dataset == 'cifar10' or args.dataset == 'fmnist' or args.dataset == 'mnist':
        num_classes = range(10)
    elif args.dataset == 'pvoc' or args.dataset == 'cifar100':
        num_classes = range(20)
    elif args.dataset == 'coco':
        num_classes = range(92)
        num_classes = [i for i in num_classes if i not in [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]]
    else:
        raise Exception('Dataset not supported')

    aucs = []

    ### Unimodal settings ###
    for _class in num_classes:
        args.label = [_class]
        print(f"Normal labels : {args.label}")
        auc = main(args)
        aucs.append(auc)
    print("Test AUCs: ", aucs)
    print("Average test AUC: ", sum(aucs)/len(aucs))


    ### Multimodal settings ###
    # for abnormal_class in num_classes:
    #     args.label = list(num_classes)
    #     args.label.pop(abnormal_class)
    #     print(f"Normal labels {args.label} | Anomaly label {abnormal_class}")
    #     auc = main(args)
    #     aucs.append(auc)
    # print("Test AUCs: ", aucs)
    # print("Average test AUC: ", sum(aucs)/len(aucs))


