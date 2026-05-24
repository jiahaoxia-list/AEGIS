import torch
import torch.distributed as dist
import os


def detail_config(dist_need):
    if dist_need is True:
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        rank = dist.get_rank()
        import sys

        if rank == 0:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        else:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

    config = dict()
    config['model'] = {
        'name': 'RefineNet',
        'in_channels': 3,
        'out_channels': 1,
        'layer_order': 'gcr',
        'f_maps': [8, 16],
        'num_groups': 8,
        'is_segmentation': False
    }
    config['trainer'] = {
        'train_namespace': 'train',
        'val_namespace': 'val',
        'test_namespace': 'test',
        'log_path': 'log_MMF/',
        'results_path': 'YOUR_SAVE_PATH',
        'epochs': 1000,
        'callback_iters': 1
    }
    config['optimizer'] = {
        'name': 'Adam',
        'betas': (0.9, 0.999),
        'learning_rate': 0.0002,
        'weight_decay': 0.00001
    }
    config['loss'] = {
        'name': 'SmoothL1Loss'
    }
    config['metric'] = {
        'name': ['PSNR', 'SSIM', 'MSE'],
        'save_by': 'PSNR'
    }
    config['dataset'] = {
        'path': 'YOUR_DATASET_ROOT',
        'test_path': 'YOUR_TEST_DATASET',
        'save_path': 'YOUR_SAVE_PATH',
        'spacing': (0.357421875, 0.357421875, 0.357421875),
        'batch_size': 4,
        'shuffle': True,
        'slice': True,
        'patch_size': (128, 512, 512),
        'stride_size': (32, 512, 512),
        'split_ratio': (0.7, 0.1, 0.2),
        'cache_num': 4
    }

    return config
