import argparse
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
def main():
    parser = argparse.ArgumentParser(description='Deep SVDD anomaly detection')
    parser.add_argument('--dataset_name', type=str, choices=['mnist', 'cifar10', 'pvoc'], default='pvoc', help='Name of the dataset to load.')
    parser.add_argument('--net_name', type=str, choices=['mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU','pvoc_LeNet'], default='pvoc_LeNet', help='Name of the neural network to use.')
    parser.add_argument('--xp_path', type=str, default='./log',help='Export path for logging the experiment.')
    parser.add_argument('--data_path', type=str, default='./VOCSegmentation', help='Root path of data.')
    parser.add_argument('--load_config', type=str, default=None, help='Config JSON-file path (default: None).')
    parser.add_argument('--load_model', type=str, default=None, help='Model file path (default: None).')
    parser.add_argument('--objective', type=str, choices=['one-class', 'soft-boundary'], default='one-class', help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
    parser.add_argument('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
    parser.add_argument('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
    parser.add_argument('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
    parser.add_argument('--optimizer_name', type=str, choices=['adam', 'amsgrad'], default='adam', help='Name of the optimizer to use for Deep SVDD network training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for Deep SVDD network training. Default=0.001')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr_milestone', type=list, default=[20,40] ,nargs='+', help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
    parser.add_argument('--pretrain', type=bool, default=True, help='Pretrain neural network parameters via autoencoder.')
    parser.add_argument('--ae_optimizer_name', type=str, choices=['adam', 'amsgrad'], default='adam', help='Name of the optimizer to use for autoencoder pretraining.')
    parser.add_argument('--ae_lr', type=float, default=0.001, help='Initial learning rate for autoencoder pretraining. Default=0.001')
    parser.add_argument('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
    parser.add_argument('--ae_lr_milestone', type=int, default=[30,80], nargs='+', help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
    parser.add_argument('--ae_batch_size', type=int, default=120, help='Batch size for mini-batch autoencoder training.')
    parser.add_argument('--ae_weight_decay', type=float, default=1e-6, help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
    parser.add_argument('--n_jobs_dataloader', type=int, default=0, help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
    # parser.add_argument('--normal_class', type=int, default=17, help='Specify the normal class of the dataset (all other classes are considered anomalous).')
    args = parser.parse_args()


    ### Unimodal Settings ###

    for normal_class in range(20):

        args.normal_class = [normal_class]
        # Get configuration
        cfg = Config(vars(args))

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = args.xp_path + '/log.txt'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Print arguments
        logger.info('Log file is %s.' % log_file)
        logger.info('Data path is %s.' % args.data_path)
        logger.info('Export path is %s.' % args.xp_path)

        logger.info('Dataset: %s' % args.dataset_name)
        logger.info(f"Normal class: {args.normal_class}")
        logger.info('Network: %s' % args.net_name)

        # If specified, load experiment config from JSON-file
        if args.load_config:
            cfg.load_config(import_json=args.load_config)
            logger.info('Loaded configuration from %s.' % args.load_config)

        # Print configuration
        logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
        logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

        # Set seed
        if cfg.settings['seed'] != -1:
            random.seed(cfg.settings['seed'])
            np.random.seed(cfg.settings['seed'])
            torch.manual_seed(cfg.settings['seed'])
            logger.info('Set seed to %d.' % cfg.settings['seed'])

        # Default device to 'cpu' if cuda is not available
        if not torch.cuda.is_available():
            args.device = 'cpu'
        logger.info('Computation device: %s' % args.device)
        logger.info('Number of dataloader workers: %d' % args.n_jobs_dataloader)

        # Load data
        dataset = load_dataset(args.dataset_name, args.data_path, args.normal_class)

        # Initialize DeepSVDD model and set neural network \phi
        deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
        deep_SVDD.set_network(args.net_name)
        # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
        if args.load_model:
            deep_SVDD.load_model(model_path=args.load_model, load_ae=True)
            logger.info('Loading model from %s.' % args.load_model)

        logger.info('Pretraining: %s' % args.pretrain)
        if args.pretrain:
            # Log pretraining details
            logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
            logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
            logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
            logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
            logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
            logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

            # Pretrain model on dataset (via autoencoder)
            deep_SVDD.pretrain(dataset,
                            optimizer_name=cfg.settings['ae_optimizer_name'],
                            lr=cfg.settings['ae_lr'],
                            n_epochs=cfg.settings['ae_n_epochs'],
                            lr_milestones=cfg.settings['ae_lr_milestone'],
                            batch_size=cfg.settings['ae_batch_size'],
                            weight_decay=cfg.settings['ae_weight_decay'],
                            device=args.device,
                            n_jobs_dataloader=args.n_jobs_dataloader)

        # Log training details
        logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
        logger.info('Training learning rate: %g' % cfg.settings['lr'])
        logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
        logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
        logger.info('Training batch size: %d' % cfg.settings['batch_size'])
        logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

        # Train model on dataset
        deep_SVDD.train(dataset,
                        optimizer_name=cfg.settings['optimizer_name'],
                        lr=cfg.settings['lr'],
                        n_epochs=cfg.settings['n_epochs'],
                        lr_milestones=cfg.settings['lr_milestone'],
                        batch_size=cfg.settings['batch_size'],
                        weight_decay=cfg.settings['weight_decay'],
                        device=args.device,
                        n_jobs_dataloader=args.n_jobs_dataloader)

        # Test model
        deep_SVDD.test(dataset, device=args.device, n_jobs_dataloader=args.n_jobs_dataloader)

        # Save results, model, and configuration
        deep_SVDD.save_results(export_json=args.xp_path + '/results.json')
        deep_SVDD.save_model(export_model=args.xp_path + '/model.tar')
        cfg.save_config(export_json=args.xp_path + '/config.json')


    ### Multimodal settings ###

    # for abnormal_class in range(20):

    #     normal_class = list(range(20))
    #     normal_class.remove(abnormal_class)
    #     args.normal_class = normal_class

    #     # Get configuration
    #     cfg = Config(vars(args))

    #     # Set up logging
    #     logging.basicConfig(level=logging.INFO)
    #     logger = logging.getLogger()
    #     logger.setLevel(logging.INFO)
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     log_file = args.xp_path + '/log.txt'
    #     file_handler = logging.FileHandler(log_file)
    #     file_handler.setLevel(logging.INFO)
    #     file_handler.setFormatter(formatter)
    #     logger.addHandler(file_handler)

    #     # Print arguments
    #     logger.info('Log file is %s.' % log_file)
    #     logger.info('Data path is %s.' % args.data_path)
    #     logger.info('Export path is %s.' % args.xp_path)

    #     logger.info('Dataset: %s' % args.dataset_name)
    #     logger.info(f"Normal class: {args.normal_class}")
    #     logger.info('Network: %s' % args.net_name)

    #     # If specified, load experiment config from JSON-file
    #     if args.load_config:
    #         cfg.load_config(import_json=args.load_config)
    #         logger.info('Loaded configuration from %s.' % args.load_config)

    #     # Print configuration
    #     logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    #     logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    #     # Set seed
    #     if cfg.settings['seed'] != -1:
    #         random.seed(cfg.settings['seed'])
    #         np.random.seed(cfg.settings['seed'])
    #         torch.manual_seed(cfg.settings['seed'])
    #         logger.info('Set seed to %d.' % cfg.settings['seed'])

    #     # Default device to 'cpu' if cuda is not available
    #     if not torch.cuda.is_available():
    #         args.device = 'cpu'
    #     logger.info('Computation device: %s' % args.device)
    #     logger.info('Number of dataloader workers: %d' % args.n_jobs_dataloader)

    #     # Load data
    #     dataset = load_dataset(args.dataset_name, args.data_path, args.normal_class)

    #     # Initialize DeepSVDD model and set neural network \phi
    #     deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    #     deep_SVDD.set_network(args.net_name)
    #     # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    #     if args.load_model:
    #         deep_SVDD.load_model(model_path=args.load_model, load_ae=True)
    #         logger.info('Loading model from %s.' % args.load_model)

    #     logger.info('Pretraining: %s' % args.pretrain)
    #     if args.pretrain:
    #         # Log pretraining details
    #         logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
    #         logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
    #         logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
    #         logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
    #         logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
    #         logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

    #         # Pretrain model on dataset (via autoencoder)
    #         deep_SVDD.pretrain(dataset,
    #                         optimizer_name=cfg.settings['ae_optimizer_name'],
    #                         lr=cfg.settings['ae_lr'],
    #                         n_epochs=cfg.settings['ae_n_epochs'],
    #                         lr_milestones=cfg.settings['ae_lr_milestone'],
    #                         batch_size=cfg.settings['ae_batch_size'],
    #                         weight_decay=cfg.settings['ae_weight_decay'],
    #                         device=args.device,
    #                         n_jobs_dataloader=args.n_jobs_dataloader)

    #     # Log training details
    #     logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    #     logger.info('Training learning rate: %g' % cfg.settings['lr'])
    #     logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    #     logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    #     logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    #     logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    #     # Train model on dataset
    #     deep_SVDD.train(dataset,
    #                     optimizer_name=cfg.settings['optimizer_name'],
    #                     lr=cfg.settings['lr'],
    #                     n_epochs=cfg.settings['n_epochs'],
    #                     lr_milestones=cfg.settings['lr_milestone'],
    #                     batch_size=cfg.settings['batch_size'],
    #                     weight_decay=cfg.settings['weight_decay'],
    #                     device=args.device,
    #                     n_jobs_dataloader=args.n_jobs_dataloader)

    #     # Test model
    #     deep_SVDD.test(dataset, device=args.device, n_jobs_dataloader=args.n_jobs_dataloader)

    #     # Save results, model, and configuration
    #     deep_SVDD.save_results(export_json=args.xp_path + '/results.json')
    #     deep_SVDD.save_model(export_model=args.xp_path + '/model.tar')
    #     cfg.save_config(export_json=args.xp_path + '/config.json')


if __name__ == '__main__':
    main()
