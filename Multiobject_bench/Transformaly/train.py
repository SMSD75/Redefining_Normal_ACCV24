import argparse
import logging
import pickle
import os
from data import Coco_Handler, PascalVOCDataModule
import torch.nn
from utils import *
from os.path import join
from pytorch_pretrained_vit.model import AnomalyViT, ViT
from timm.models.vision_transformer import vit_base_patch16_224
import numpy as np
import torchvision.transforms as transforms

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transformaly')
    parser.add_argument('--dataset',default='pvoc')
    parser.add_argument('--data_path', default='./Transformaly_datasets/', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--lr', default=0.0001,
                        help='Learning rate value')
    parser.add_argument('--eval_every', type=int, default=2,
                        help='Will evaluate the model ever <eval_every> epochs')
    parser.add_argument('--unimodal', default=False, action='store_true',
                        help='Use the unimodal settings')
    parser.add_argument('--plot_every_layer_summarization', default=False, action='store_true',
                        help='plot the per layer AUROC')
    parser.add_argument('--mode', default='trained', choices=['trained', 'pretrained'], help='Use the trained or pretrained model')
    parser_args = parser.parse_args()
    args = vars(parser_args)


    args['use_layer_outputs'] = list(range(2, 12))
    args['use_imagenet'] = False
    BASE_PATH = 'experiments'

    if args['dataset'] == 'cifar10':
        _classes = range(10)
    elif args['dataset'] == 'fmnist':
        _classes = range(10)
    elif args['dataset'] == 'cifar100':
        _classes = range(20)
    elif args['dataset'] == 'cats_vs_dogs':
        _classes = range(2)
    elif args['dataset'] == 'dior':
        _classes = range(19)
    elif args['dataset'] == 'pvoc':
        _classes = range(20)
    elif args['dataset'] == 'coco':
        _classes = range(92)
        _classes = [i for i in _classes if i not in [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]]
    else:
        raise ValueError(f"Does not support the {args['dataset']} dataset")
    # create the relevant directories
    if not os.path.exists(
            join(BASE_PATH,
                 f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}')):
        os.makedirs(join(BASE_PATH,
                         f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}'))

    logging.basicConfig(
        filename=join(BASE_PATH,
                      f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}',
                      f'Train_{args["dataset"]}_Transformaly_outputs.log'), level=logging.DEBUG)

    print_and_add_to_log("========================================================", logging)
    print_and_add_to_log("Args are:", logging)
    print_and_add_to_log(args, logging)
    print_and_add_to_log("========================================================", logging)

    for _class in _classes:
        print_and_add_to_log("===================================", logging)
        print_and_add_to_log(f"Class is : {_class}", logging)
        print_and_add_to_log("===================================", logging)
        args['_class'] = _class
        base_feature_path = join(
            BASE_PATH,
            f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}/class_{_class}')
        model_path = join(base_feature_path, 'model')

        args['base_feature_path'] = base_feature_path
        args['model_path'] = model_path

        # create the relevant directories
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if args['unimodal']:
            anomaly_classes = [i for i in _classes if i != args['_class']]
        else:
            anomaly_classes = [args['_class']]

        print_and_add_to_log(
            "====================================================================",
            logging)
        print_and_add_to_log(
            "Start Training", logging)
        print_and_add_to_log(
            "====================================================================",
            logging)

        if args['dataset'] == 'coco':
        
            mod = Coco_Handler(
                batch_size=args['batch_size'], 
                normal_classes=[args['_class']], 
                transformations=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]),
                num_workers=2
                )
            testset = mod.test_dataset
            dataloaders = {'training': mod.get_train_loader(),
                        'val': mod.get_test_loader_normal(),
                        'test': mod.get_test_loader_ood(),
                        'overall_test': mod.get_test_loader(),
                        }   

        elif args['dataset'] == 'pvoc':
         
            mod = PascalVOCDataModule(
                batch_size=args['batch_size'], 
                normal_classes=[args['_class']], 
                image_train_transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
                )
            testset = mod.test_dataset
            dataloaders = {'training': mod.get_train_dataloader(),
                        'val': mod.get_normal_test_dataloader(),
                        'test': mod.get_ood_test_dataloader(),
                        'overall_test': mod.get_test_dataloader(),
                        }

        else:

            trainset, testset = get_datasets_for_ViT(dataset=args['dataset'], data_path=args['data_path'], one_vs_rest=args['unimodal'], _class=args['_class'], normal_test_sample_only=True, use_imagenet=args['use_imagenet'])
            _, ood_test_set = get_datasets_for_ViT(dataset=args['dataset'], data_path=args['data_path'], one_vs_rest=not args['unimodal'], _class=args['_class'], normal_test_sample_only=True, use_imagenet=args['use_imagenet'])

            print_and_add_to_log("---------------", logging)
            print_and_add_to_log(f'Class size: {args["_class"]}', logging)
            print_and_add_to_log(f'Trainset size: {len(trainset)}', logging)
            print_and_add_to_log(f'Testset size: {len(testset)}', logging)
            print_and_add_to_log(f'OOD testset size: {len(ood_test_set)}', logging)
            print_and_add_to_log("---------------", logging)

            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'],
                                                    shuffle=True)
            val_loader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'],
                                                    shuffle=False)
            ood_val_loader = torch.utils.data.DataLoader(ood_test_set, batch_size=args['batch_size'],
                                                        shuffle=False)
            dataloaders = {'training': train_loader,
                        'val': val_loader,
                        'test': ood_val_loader
                        }

        # Build model
        if args['use_imagenet']:
            VIT_MODEL_NAME = 'B_16_imagenet1k'
        else:
            VIT_MODEL_NAME = 'B_16'

        # Build model
        model = AnomalyViT(VIT_MODEL_NAME, pretrained=True)
        model.fc = Identity()
        # Build model for best instance
        best_model = AnomalyViT(VIT_MODEL_NAME, pretrained=True)
        best_model.fc = Identity()

        model.to('cuda')
        best_model.to('cuda')

        model_checkpoint_path = join(model_path, 'last_full_finetuned_model_state_dict.pkl')
        if os.path.exists(model_checkpoint_path):
            model_state_dict = torch.load(model_checkpoint_path)
            model.load_state_dict(model_state_dict)
            print_and_add_to_log("model loadded from checkpoint here:", logging)
            print_and_add_to_log(model_checkpoint_path, logging)


        if args['mode'] == 'trained':

            freeze_finetuned_model(model)
            model, best_model, cur_acc_loss = train(model=model,
                                                    best_model=best_model,
                                                    args=args,
                                                    dataloaders=dataloaders,
                                                    output_path=model_path,
                                                    device='cuda',
                                                    seed=42,
                                                    model_checkpoint_path=model_checkpoint_path,
                                                    anomaly_classes=anomaly_classes
                                                    )

            training_losses = cur_acc_loss['training_losses']
            val_losses = cur_acc_loss['val_losses']
            try:
                plot_graphs(training_losses, val_losses, training_losses, val_losses)

            except Exception as e:
                print_and_add_to_log('raise error:', logging)
                print_and_add_to_log(e, logging)

            # save models
            torch.save(best_model.state_dict(), join(model_path,
                                                    'best_full_finetuned_model_state_dict.pkl'))
            torch.save(model.state_dict(), join(model_path,
                                                'last_full_finetuned_model_state_dict.pkl'))

            # save losses
            with open(join(model_path, 'full_finetuned_training_losses.pkl'), 'wb') as f:
                pickle.dump(training_losses, f)
            with open(join(model_path, 'full_finetuned_val_losses.pkl'), 'wb') as f:
                pickle.dump(val_losses, f)

            if args['use_imagenet']:
                MODEL_NAME = 'B_16_imagenet1k'
            else:
                MODEL_NAME = 'B_16'

            model = ViT(MODEL_NAME, pretrained=True)
            model.fc = Identity()
            model.load_state_dict(best_model.state_dict(), strict=False)

        elif args['mode'] == 'pretrained':

            if args['use_imagenet']:
                MODEL_NAME = 'B_16_imagenet1k'
            else:
                MODEL_NAME = 'B_16'

            model = ViT(MODEL_NAME, pretrained=True)
            model.fc = Identity()
            pretraining = vit_base_patch16_224(pretrained=True).to('cuda')
            model.load_state_dict(pretraining.state_dict(), strict=False)

        else:
            raise ValueError(f"Does not support the {args['mode']} mode")

        model.eval()
        model.to('cuda')
        
        extract_features(base_path=BASE_PATH,
                        data_path=args['data_path'],
                        datasets=[args['dataset']],
                        model=model,
                        logging=logging,
                        calculate_features=True,
                        unimodal_vals=[args['unimodal']],
                        output_train_features=True,
                        output_test_features=True,
                        use_imagenet=args['use_imagenet'])
