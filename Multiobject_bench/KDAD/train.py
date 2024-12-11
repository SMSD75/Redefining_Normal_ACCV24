from test import *
from utils.utils import *
from dataloader import *
from pathlib import Path
from torch.autograd import Variable
import pickle
from test_functions import detection_test
from loss_functions import *
from argparse import ArgumentParser
from models.network import get_networks

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='./KDAD/configs/config.yaml', help="training configuration")


def train(config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    direction_loss_only = config["direction_loss_only"]
    normal_class = config["normal_class"]
    learning_rate = float(config['learning_rate'])
    num_epochs = config["num_epochs"]
    lamda = config['lamda']
    continue_train = config['continue_train']
    last_checkpoint = config['last_checkpoint']

    train_dataloader, test_dataloader = load_data(config)

    checkpoint_path = "./outputs/{}/{}/checkpoints/".format(config['experiment_name'], config['dataset_name'])
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    if continue_train:
        vgg, model = get_networks(config, load_checkpoint=True)
    else:
        vgg, model = get_networks(config)

    # Criteria And Optimizers
    if direction_loss_only:
        criterion = DirectionOnlyLoss()
    else:
        criterion = MseDirectionLoss(lamda)

    vgg = vgg.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if continue_train:
        optimizer.load_state_dict(
            torch.load('{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, last_checkpoint)))

    losses = []
    roc_aucs = []
    if continue_train:
        with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, last_checkpoint), 'rb') as f:
            roc_aucs = pickle.load(f)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for data,_ in train_dataloader:
            images = data.to(device)
            output_pred = model.forward(images)
            output_real = vgg(images)
            total_loss = criterion(output_pred, output_real)
            epoch_loss += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, epoch_loss))

        if epoch % 10 == 0:
            roc_auc = detection_test(model, vgg, test_dataloader, config)
            roc_aucs.append(roc_auc)
            print("RocAUC at epoch {}:".format(epoch+1), roc_auc)

        if epoch % 50 == 0:
            torch.save(model.state_dict(),
                        '{}Cloner_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
            torch.save(optimizer.state_dict(),
                        '{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
            with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, epoch),
                        'wb') as f:
                pickle.dump(roc_aucs, f)


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    train(config)

if __name__ == '__main__':
    main()
