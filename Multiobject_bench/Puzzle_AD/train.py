from torch import nn
from random import randrange
from models.Discriminator import *
from test import *
from dataloader import *
from pathlib import Path
from utils.utils import *
from argparse import ArgumentParser
from models.Unet import *
from models.Discriminator import *
from torch.autograd import Variable
from test import find_fpr, test, get_avg_val_error_per_permutation

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='./Puzzle_AD/configs/config_train.yaml', help="training configuration")

def main(train_dataloader, test_dataloader, target_class):

    args = parser.parse_args()
    config = get_config(args.config)

    n_channel = config['n_channel']
    normal_class = target_class #config["normal_class"]
    dataset_name = config['dataset_name']

    epsilon = float(config['eps'])
    alpha = float(config['alpha'])

    if dataset_name == 'MVTec' and get_mvtec_class_type(normal_class) == 'texture':
        get_random_permutation = get_forced_random_permutation
    else:
        get_random_permutation = get_unforced_random_permutation


    unet = UNet(n_channel, n_channel, config['base_channel']).cuda()
    discriminator = NetD(config['image_size'], n_channel, config['n_extra_layers']).cuda()
    discriminator.apply(weights_init)

    unet.train()
    discriminator.train()

    criterion = nn.MSELoss()
    optimizer_u = torch.optim.Adam(
        unet.parameters(), lr=config['lr_u'], weight_decay=float(config['weight_decay']))

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=config['lr_d'], betas=(config['beta1'], config['beta2']))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_u, factor=config['factor'], patience=config['patience'], mode='min', verbose=True)

    ae_loss_list = []

    l_adv = l2_loss
    l_bce = nn.BCELoss()

    num_epochs = config['num_epochs']
    epoch_loss_dict = dict()
    unet.train()
    discriminator.train()

    for epoch in range(num_epochs):
        epoch_ae_loss = 0
        epoch_total_loss = 0

        for data in train_dataloader:
            rand_number = randrange(4)

            img = data
            orig_img = img

            partitioned_img, base = split_tensor(img, tile_size=img.size(2) // 2, offset=img.size(2) // 2)
            perm = get_random_permutation()

            extended_perm = perm * img.size(1)
            if img.size(1) == 3:
                offset = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
                final_perm = offset + extended_perm[:, None]
                final_perm = final_perm.view(-1)
            else:
                final_perm = extended_perm

            permuted_img = partitioned_img[:, final_perm, :, :]

            if img.size(1) == 3:
                avg = permuted_img[:, rand_number * 3, :, :] + permuted_img[:, rand_number * 3 + 1, :, :] + \
                      permuted_img[:, rand_number * 3 + 2, :, :]

                avg /= 3
                permuted_img[:, rand_number * 3, :, :] = avg
                permuted_img[:, rand_number * 3 + 1, :, :] = avg
                permuted_img[:, rand_number * 3 + 2, :, :] = avg
            else:
                permuted_img[:, rand_number, :, :] *= 0

            target = orig_img
            permuted_img = rebuild_tensor(permuted_img, base, tile_size=img.size(2) // 2, offset=img.size(2) // 2)

            permuted_img = fgsm_attack(permuted_img, unet, eps=epsilon, alpha=alpha)

            img = Variable(permuted_img).cuda()
            target = Variable(target).cuda()

            # ===================forward=====================

            # Forward Unet
            output = unet(img)

            # Forward Discriminator
            pred_real, feat_real = discriminator(target)
            pred_fake, feat_fake = discriminator(output.detach())

            # ===================backward====================

            # Backward Unet
            optimizer_u.zero_grad()
            err_g_adv = l_adv(discriminator(target)[1], discriminator(output)[1])
            AE_loss = criterion(output, target)
            loss = config['adv_coeff'] * err_g_adv + AE_loss

            epoch_total_loss += loss.item()
            epoch_ae_loss += AE_loss.item()
            loss.backward()
            optimizer_u.step()

            # Backward Discriminator
            real_label = torch.ones(size=(16*img.shape[0],), dtype=torch.float32).cuda() # Size = Batch_size x 1 -> 32x1
            fake_label = torch.zeros(size=(16*img.shape[0],), dtype=torch.float32).cuda()
            optimizer_d.zero_grad()
            err_d_real = l_bce(pred_real, real_label) # Pred_real shape = Batch_size x 1 x 4 x 4 -> 512x1
            err_d_fake = l_bce(pred_fake, fake_label)

            err_d = (err_d_real + err_d_fake) * 0.5
            err_d.backward()
            optimizer_d.step()

        # ===================log========================
        ae_loss_list.append(epoch_ae_loss)
        scheduler.step(epoch_ae_loss)

        print(f"epoch [{epoch+1}/{num_epochs}], epoch_total_loss:{epoch_total_loss:.4f}, epoch_ae_loss:{epoch_ae_loss:.4f}")

        # with open(checkpoint_path + 'log_{}.txt'.format(normal_class), "a") as log_file:
        #     log_file.write('\n epoch [{}/{}], loss:{:.4f}, epoch_ae_loss:{:.4f}, err_adv:{:.4f}'
        #                    .format(epoch + 1, num_epochs, epoch_total_loss, epoch_ae_loss, err_g_adv))

        # if epoch % 500 == 0:
        #     show_process_for_trainortest(img, output, orig_img, train_output_path + str(epoch))
        #     epoch_loss_dict[epoch] = epoch_total_loss

        #     torch.save(unet.state_dict(), checkpoint_path + '{}.pth'.format(str(epoch)))
        #     torch.save(discriminator.state_dict(), checkpoint_path + 'netd_{}.pth'.format(str(epoch)))

        #     torch.save(optimizer_u.state_dict(), checkpoint_path + 'opt_{}.pth'.format(str(epoch)))
        #     torch.save(optimizer_d.state_dict(), checkpoint_path + 'optd_{}.pth'.format(str(epoch)))

        #     torch.save(scheduler.state_dict(), checkpoint_path + 'scheduler_{}.pth'.format(str(epoch)))

    # TEST
    model = unet
    model.eval()
    permutation_list = get_all_permutations()
    perm_cost = get_avg_val_error_per_permutation(model, permutation_list, test_dataloader)
    target_class = normal_class
    auc_dict = test(model, target_class, permutation_list, perm_cost, test_dataloader)
    print(auc_dict)

    return auc_dict



if __name__ == '__main__':
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    test_aucs = []

    for normal_class in range(20):

        print(f"Normal Class : {normal_class}")
        args = parser.parse_args()
        config = get_config(args.config)

        img_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
            ])

        normal_classes = [normal_class]

        mod = PascalVOCDataModule(batch_size=config['batch_size'], train_transform=img_transform, val_transform=img_transform, test_transform=img_transform, normal_classes = normal_classes)
        train_dataloader = mod.get_train_dataloader()
        test_dataloader = mod.get_test_dataloader()

        test_auc = main(train_dataloader, test_dataloader, normal_class)
        test_aucs.append(test_auc)

    print(f"Test AUCs: {test_aucs}")