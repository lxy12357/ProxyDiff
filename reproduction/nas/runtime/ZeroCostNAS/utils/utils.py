from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
import shutil
import sys
import logging
import argparse
import torchvision.datasets as dset
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from sklearn import metrics
from scipy import stats
from torch.autograd import Variable

from collections import OrderedDict

import random
import os
import fcntl
import os.path
import gzip
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

from fvcore.common.checkpoint import Checkpointer as fvCheckpointer
from fvcore.common.config import CfgNode

# Taskonomy utilities are intentionally not imported here. They pull optional
# 3D data dependencies that are not used by the CIFAR/NASBench reproduction.

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


cat_channels = partial(torch.cat, dim=1)

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Returns the root path of the project.
    """
    return Path(__file__).parent.parent


def iter_flatten(iterable):
    """
    Flatten a potentially deeply nested python list
    """
    # taken from https://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in iter_flatten(e):
                yield f
        else:
            yield e


def default_argument_parser():
    """
    Returns the argument parser with the default options.

    Inspired by the implementation of FAIR's detectron2
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default=None, metavar="FILE", help="Path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--datapath", default=None, metavar="FILE", help="Path to the folder with train/test data folders")
    return parser


def parse_args(parser=default_argument_parser(), args=sys.argv[1:]):
    if "-f" in args:
        args = args[2:]
    return parser.parse_args(args)


def pairwise(iterable):
    """
    Iterate pairwise over list.

    from https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    """
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def load_config(path):
    with open(path) as f:
        config = CfgNode.load_cfg(f)

    return config

def load_default_config():
    config_paths = "configs/predictor_config.yaml"

    config_path_full = os.path.join(
        *(
            [get_project_root()] + config_paths.split('/')
        )
    )

    return load_config(config_path_full)

def get_config_from_args(args=None):
    """
    Parses command line arguments and merges them with the defaults
    from the config file.

    Prepares experiment directories.

    Args:
        args: args from a different argument parser than the default one.
    """

    if args is None:
        args = parse_args()
    else:
        args = parse_args(args)
    logger.info("Command line args: {}".format(args))

    if args.config_file is None:
        config = load_default_config()
    else:
        config = load_config(path=args.config_file)

    # Override file args with ones from command line
    try:
        for arg, value in pairwise(args.opts):
            if "." in arg:
                arg1, arg2 = arg.split(".")
                config[arg1][arg2] = type(config[arg1][arg2])(value)
            else:
                if arg in config:
                    t = type(config[arg])
                elif value.isnumeric():
                    t = int
                else:
                    t = str
                config[arg] = t(value)

        # load config file
        config.set_new_allowed(True)
        config.merge_from_list(args.opts)

    except AttributeError:
        for arg, value in pairwise(args):
            config[arg] = value

    if args.datapath is not None:
        # logger.info(args.datapath)
        config.train_data_file = os.path.join(args.datapath, 'train.json')
        config.test_data_file = os.path.join(args.datapath, 'test.json')
    else:
        config.train_data_file = None
        config.test_data_file = None

    # prepare the output directories
    config.save = "{}/{}/{}/{}/{}/{}".format(
        config.out_dir,
        config.config_type,
        config.search_space,
        config.dataset,
        config.predictor,
        config.seed,
    )
    config.data = os.environ.get(
        "CIFAR10_DATA_ROOT", "{}/data".format(get_project_root())
    )

    create_exp_dir(config.save)
    create_exp_dir(config.save + "/search")  # required for the checkpoints
    create_exp_dir(config.save + "/eval")

    return config

def load_ninapro(path, whichset):
    data_str = 'ninapro_' + whichset + '.npy'
    label_str = 'label_' + whichset + '.npy'

    data = np.load(os.path.join(path, data_str),
                             encoding="bytes", allow_pickle=True)
    labels = np.load(os.path.join(path, label_str), encoding="bytes", allow_pickle=True)

    data = np.transpose(data, (0, 2, 1))
    data = data[:, None, :, :]
    data = torch.from_numpy(data.astype(np.float32))
    labels = torch.from_numpy(labels.astype(np.int64))

    all_data = TensorDataset(data, labels)
    return all_data

def get_train_val_loaders(config,for_train=False,split_mode=0,fix_order_valid=False,auto_augment=False,batch_size=None,batch_size_valid=None):
    """
    Constructs the dataloaders and transforms for training, validation and test data.
    """
    data = config.data
    dataset = config.dataset
    seed = config.seed
    val_data = None
    if dataset == "cifar10":
        train_transform, valid_transform = _data_transforms_cifar10(config,auto_augment)
        os.makedirs(data, exist_ok=True)
        with open(os.path.join(data, ".cifar10_download.lock"), "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            train_data = dset.CIFAR10(
                root=data, train=True, download=True, transform=train_transform
            )
            val_data = dset.CIFAR10(
                root=data, train=True, download=True, transform=valid_transform
            )
            test_data = dset.CIFAR10(
                root=data, train=False, download=True, transform=valid_transform
            )
    elif dataset == "cifar100":
        train_transform, valid_transform = _data_transforms_cifar100(config)
        train_data = dset.CIFAR100(
            root=data, train=True, download=True, transform=train_transform
        )
        val_data = dset.CIFAR100(
            root=data, train=True, download=True, transform=valid_transform
        )
        test_data = dset.CIFAR100(
            root=data, train=False, download=True, transform=valid_transform
        )
    elif dataset == "svhn":
        train_transform, valid_transform = _data_transforms_svhn(config)
        train_data = dset.SVHN(
            root=data, split="train", download=True, transform=train_transform
        )
        test_data = dset.SVHN(
            root=data, split="test", download=True, transform=valid_transform
        )
    elif dataset == "ImageNet16-120":
        from ZeroCostNAS.utils.DownsampledImageNet import ImageNet16

        train_transform, valid_transform = _data_transforms_ImageNet_16_120(config)
        data_folder = f"{data}/{dataset}"
        train_data = ImageNet16(
            root=data_folder,
            train=True,
            transform=train_transform,
            use_num_of_class_only=120,
        )
        val_data = ImageNet16(
            root=data_folder,
            train=True,
            transform=valid_transform,
            use_num_of_class_only=120,
        )
        test_data = ImageNet16(
            root=data_folder,
            train=False,
            transform=valid_transform,
            use_num_of_class_only=120,
        )
    elif dataset == "scifar100":
        data_file = os.path.join(data, 's2_cifar100.gz')
        with gzip.open(data_file, 'rb') as f:
            dataset = pickle.load(f)

        train_img = torch.from_numpy(
            dataset["train"]["images"][:, :, :, :].astype(np.float32))
        train_labels = torch.from_numpy(
            dataset["train"]["labels"].astype(np.int64))
        train_data = TensorDataset(train_img, train_labels)
        test_img = torch.from_numpy(
            dataset["test"]["images"][:, :, :, :].astype(np.float32))
        test_labels = torch.from_numpy(
            dataset["test"]["labels"].astype(np.int64))
        test_data = TensorDataset(test_img, test_labels)

        train_transform, valid_transform = None, None
    elif dataset == "ninapro":
        path = os.path.join(data, 'ninapro')

        trainset = load_ninapro(path, 'train')
        valset = load_ninapro(path, 'val')
        train_data = ConcatDataset([trainset, valset])
        test_data = load_ninapro(path, 'test')

        train_transform, valid_transform = None, None
    elif dataset == 'jigsaw':
        cfg = get_jigsaw_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'class_object':
        cfg = get_class_object_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'class_scene':
        cfg = get_class_scene_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'autoencoder':
        cfg = get_autoencoder_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'segmentsemantic':
        cfg = get_segmentsemantic_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'normal':
        cfg = get_normal_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'room_layout':
        cfg = get_room_layout_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    num_train = len(train_data)
    indices = list(range(num_train))
    if split_mode==1:
        split = int(np.floor(config.train_portion_half * num_train))
    elif split_mode==2:
        split = num_train
    elif split_mode==3:
        split = int(np.floor(0.9 * num_train))
    else:
        split = int(np.floor(config.train_portion * num_train))

    if batch_size:
        batch_size_config = batch_size
    else:
        batch_size_config = config.batch_size
        if for_train:
            batch_size_config = config.batch_size_train

    if batch_size_valid:
        batch_size_config_valid = batch_size_valid
    else:
        batch_size_config_valid = batch_size_config

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size_config,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        # shuffle=True,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed+1),
    )

    if fix_order_valid:
        if split_mode == 2:
            valid_queue = torch.utils.data.DataLoader(
                val_data,
                batch_size=batch_size_config_valid,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
                worker_init_fn=np.random.seed(seed + 1),
            )
        else:
            val_data = torch.utils.data.Subset(val_data,indices[split:num_train])
            valid_queue = torch.utils.data.DataLoader(
                val_data,
                batch_size=batch_size_config_valid,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
                worker_init_fn=np.random.seed(seed + 1),
            )
    else:
        valid_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size_config_valid,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            # shuffle=True,
            pin_memory=True,
            num_workers=0,
            worker_init_fn=np.random.seed(seed),
        )

    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size_config,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    return train_queue, valid_queue, test_queue, train_transform, valid_transform


def _data_transforms_cifar10(args, auto_augment):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    if auto_augment:
        random_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        random_transform.append(CIFAR10Policy())
        norm_transform = [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ]
        # if args.cutout:
        norm_transform.append(Cutout(args.cutout_length))
        train_transform = transforms.Compose(
            random_transform + norm_transform)

    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        if hasattr(args, 'cutout') and args.cutout == True:
            train_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_svhn(args):
    SVHN_MEAN = [0.4377, 0.4438, 0.4728]
    SVHN_STD = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    if hasattr(args, 'cutout') and args.cutout == True:
        train_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if hasattr(args, 'cutout') and args.cutout == True:
        train_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_ImageNet_16_120(args):
    IMAGENET16_MEAN = [x / 255 for x in [122.68, 116.66, 104.01]]
    IMAGENET16_STD = [x / 255 for x in [63.22, 61.26, 65.09]]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(16, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD),
        ]
    )
    if hasattr(args, 'cutout') and args.cutout == True:
        train_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD),
        ]
    )
    return train_transform, valid_transform


def get_jigsaw_configs():
    cfg = {}

    cfg['task_name'] = 'jigsaw'

    cfg['input_dim'] = (255, 255)
    cfg['target_num_channels'] = 9

    cfg['dataset_dir'] = os.path.join(get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(get_project_root(), "data", "final5K_splits")

    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'

    cfg['target_dim'] = 1000
    cfg['target_load_fn'] = load_ops.random_jigsaw_permutation
    cfg['target_load_kwargs'] = {'classes': cfg['target_dim']}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.RandomGrayscale(0.3),
        load_ops.MakeJigsawPuzzle(classes=cfg['target_dim'], mode='max', tile_dim=(64, 64), centercrop=0.9, norm=False, totensor=True),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.RandomGrayscale(0.3),
        load_ops.MakeJigsawPuzzle(classes=cfg['target_dim'], mode='max', tile_dim=(64, 64), centercrop=0.9, norm=False, totensor=True),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.RandomGrayscale(0.3),
        load_ops.MakeJigsawPuzzle(classes=cfg['target_dim'], mode='max', tile_dim=(64, 64), centercrop=0.9, norm=False, totensor=True),
    ])
    return cfg


def get_class_object_configs():
    cfg = {}

    cfg['task_name'] = 'class_object'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['dataset_dir'] = os.path.join(get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(get_project_root(), "data", "final5K_splits")

    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'

    cfg['target_dim'] = 75

    cfg['target_load_fn'] = load_ops.load_class_object_logits

    cfg['target_load_kwargs'] = {'selected': True if cfg['target_dim'] < 1000 else False, 'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}

    cfg['demo_kwargs'] = {'selected': True if cfg['target_dim'] < 1000 else False, 'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}

    cfg['normal_params'] = {'mean': [0.5224, 0.5222, 0.5221], 'std': [0.2234, 0.2235, 0.2236]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
       load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg


def get_class_scene_configs():
    cfg = {}

    cfg['task_name'] = 'class_scene'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['dataset_dir'] = os.path.join(get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(get_project_root(), "data", "final5K_splits")

    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'

    cfg['target_dim'] = 47

    cfg['target_load_fn'] = load_ops.load_class_scene_logits

    cfg['target_load_kwargs'] = {'selected': True if cfg['target_dim'] < 365 else False, 'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}

    cfg['demo_kwargs'] = {'selected': True if cfg['target_dim'] < 365 else False, 'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}

    cfg['normal_params'] = {'mean': [0.5224, 0.5222, 0.5221], 'std': [0.2234, 0.2235, 0.2236]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
       load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg


def get_autoencoder_configs():

    cfg = {}

    cfg['task_name'] = 'autoencoder'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['target_dim'] = (256, 256)
    cfg['target_channel'] = 3

    cfg['dataset_dir'] = os.path.join(get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(get_project_root(), "data", "final5K_splits")

    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'

    cfg['target_load_fn'] = load_ops.load_raw_img_label
    cfg['target_load_kwargs'] = {}

    cfg['normal_params'] = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg

def get_segmentsemantic_configs():

    cfg = {}

    cfg['task_name'] = 'segmentsemantic'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['target_dim'] = (256, 256)
    cfg['target_num_channel'] = 17

    cfg['dataset_dir'] = os.path.join(get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(get_project_root(), "data", "final5K_splits")

    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'

    cfg['target_load_fn'] = load_ops.semantic_segment_label
    cfg['target_load_kwargs'] = {}

    cfg['normal_params'] = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg

def get_normal_configs():

    cfg = {}

    cfg['task_name'] = 'normal'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['target_dim'] = (256, 256)
    cfg['target_channel'] = 3

    cfg['dataset_dir'] = os.path.join(get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(get_project_root(), "data", "final5K_splits")

    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'

    cfg['target_load_fn'] = load_ops.load_raw_img_label
    cfg['target_load_kwargs'] = {}

    cfg['normal_params'] = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        # load_ops.RandomHorizontalFlip(0.5),
        # load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg

def get_room_layout_configs():

    cfg = {}

    cfg['task_name'] = 'room_layout'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['target_dim'] = 9

    cfg['dataset_dir'] = os.path.join(get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(get_project_root(), "data", "final5K_splits")

    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'

    cfg['target_load_fn'] = load_ops.point_info2room_layout
    # cfg['target_load_fn'] = load_ops.room_layout
    cfg['target_load_kwargs'] = {}

    cfg['normal_params'] = {'mean': [0.5224, 0.5222, 0.5221], 'std': [0.2234, 0.2235, 0.2236]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        # load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg



class TensorDatasetWithTrans(Dataset):
    """
    TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def set_seed(seed):
    """
    Set the seeds for all used libraries.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def get_last_checkpoint(config, search=True):
    """
    Finds the latest checkpoint in the experiment directory.

    Args:
        config (AttrDict): The config from config file.
        search (bool): Search or evaluation checkpoint

    Returns:
        (str): The path to the latest checkpoint file.
    """
    try:
        path = os.path.join(
            config.save, "search" if search else "eval", "last_checkpoint"
        )
        with open(path, "r") as f:
            checkpoint_name = f.readline()
        return os.path.join(
            config.save, "search" if search else "eval", checkpoint_name
        )
    except:
        return ""


def accuracy(output, target, topk=(1,)):
    """
    Calculate the accuracy given the softmax output and the target.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_class_object(output, target, topk=(1,)):
    """
    Calculate the accuracy given the softmax output and the target.
    """
    target = target.argmax(dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_class_scene(output, target, topk=(1,)):
    """
    Calculate the accuracy given the softmax output and the target.
    """
    target = target.argmax(dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_autoencoder(output, target, topk=(1,)):
    ssim_loss= SSIM(data_range=1, size_average=True, channel=3)
    res = ssim_loss(output, target)
    """
    Calculate the accuracy given the softmax output and the target.
    """
    return res, res


def count_parameters_in_MB(model):
    """
    Returns the model parameters in mega byte.
    """
    return (
        np.sum(
            np.prod(v.size())
            for name, v in model.named_parameters()
            if "auxiliary" not in name
        )
        / 1e6
    )


# Calculate the BR@K, WR@K
def minmax_n_at_k(predict_scores, true_scores, ks=[1, 5, 10, 20, 25, 30, 50, 75, 100]):
    true_scores = np.array(true_scores)
    predict_scores = np.array(predict_scores)
    num_archs = len(true_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    predict_best_inds = np.argsort(predict_scores)[::-1]
    minn_at_ks = []
    for k in ks:
        ranks = true_ranks[predict_best_inds[:k]]
        if len(ranks) < 1:
            continue
        minn = int(np.min(ranks)) + 1
        maxn = int(np.max(ranks)) + 1
        minn_at_ks.append((k, k, minn, float(minn) / num_archs, maxn, float(maxn) / num_archs))
    return minn_at_ks


# Calculate the P@topK, P@bottomK, and Kendall-Tau in predicted topK/bottomK
def p_at_tb_k(predict_scores, true_scores, ks=[1, 5, 10, 20, 25, 30, 50, 75, 100]):
# ratios=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]):
    predict_scores = np.array(predict_scores)
    true_scores = np.array(true_scores)
    predict_inds = np.argsort(predict_scores)[::-1]
    num_archs = len(predict_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    patks = []
    for k in ks:
        # k = int(num_archs * ratio)
        if k < 1:
            continue
        top_inds = predict_inds[:k]
        bottom_inds = predict_inds[num_archs-k:]
        p_at_topk = len(np.where(true_ranks[top_inds] < k)[0]) / float(k)
        p_at_bottomk = len(np.where(true_ranks[bottom_inds] >= num_archs - k)[0]) / float(k)
        kd_at_topk = stats.kendalltau(predict_scores[top_inds], true_scores[top_inds]).correlation
        kd_at_bottomk = stats.kendalltau(predict_scores[bottom_inds], true_scores[bottom_inds]).correlation
        # [ratio, k, P@topK, P@bottomK, KT in predicted topK, KT in predicted bottomK]
        patks.append((k/len(true_scores), k, p_at_topk, p_at_bottomk, kd_at_topk, kd_at_bottomk))
    return patks



def log_args(args):
    """
    Log the args in a nice way.
    """
    for arg, val in args.items():
        logger.info(arg + "." * (50 - len(arg) - len(str(val))) + str(val))


def create_exp_dir(path):
    """
    Create the experiment directories.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    logger.info("Experiment dir : {}".format(path))


def cross_validation(
    xtrain, ytrain, predictor, split_indices, score_metric="kendalltau"
):
    validation_score = []

    for train_indices, validation_indices in split_indices:
        xtrain_i = [xtrain[j] for j in train_indices]
        ytrain_i = [ytrain[j] for j in train_indices]
        xval_i = [xtrain[j] for j in train_indices]
        yval_i = [ytrain[j] for j in train_indices]

        predictor.fit(xtrain_i, ytrain_i)
        ypred_i = predictor.query(xval_i)

        # If the predictor is an ensemble, take the mean
        if len(ypred_i.shape) > 1:
            ypred_i = np.mean(ypred_i, axis=0)

        # use Pearson correlation to be the metric -> maximise Pearson correlation
        if score_metric == "pearson":
            score_i = np.abs(np.corrcoef(yval_i, ypred_i)[1, 0])
        elif score_metric == "mae":
            score_i = np.mean(abs(ypred_i - yval_i))
        elif score_metric == "rmse":
            score_i = metrics.mean_squared_error(yval_i, ypred_i, squared=False)
        elif score_metric == "spearman":
            score_i = stats.spearmanr(yval_i, ypred_i)[0]
        elif score_metric == "kendalltau":
            score_i = stats.kendalltau(yval_i, ypred_i)[0]
        elif score_metric == "kt_2dec":
            score_i = stats.kendalltau(yval_i, np.round(ypred_i, decimals=2))[0]
        elif score_metric == "kt_1dec":
            score_i = stats.kendalltau(yval_i, np.round(ypred_i, decimals=1))[0]

        validation_score.append(score_i)

    return np.mean(validation_score)


def generate_kfold(n, k):
    """
    Input:
        n: number of training examples
        k: number of folds
    Returns:
        kfold_indices: a list of len k. Each entry takes the form
        (training indices, validation indices)
    """
    assert k >= 2
    kfold_indices = []

    indices = np.array(range(n))
    fold_size = n // k

    fold_indices = [indices[i * fold_size : (i + 1) * fold_size] for i in range(k - 1)]
    fold_indices.append(indices[(k - 1) * fold_size :])

    for i in range(k):
        training_indices = [fold_indices[j] for j in range(k) if j != i]
        validation_indices = fold_indices[i]
        kfold_indices.append((np.concatenate(training_indices), validation_indices))

    return kfold_indices


def compute_scores(ytest, test_pred):
    ytest = np.array(ytest)
    test_pred = np.array(test_pred)
    METRICS = [
        "mae",
        "rmse",
        "pearson",
        "spearman",
        "kendalltau",
        "kt_2dec",
        "kt_1dec",
        "full_ytest",
        "full_testpred",
    ]
    metrics_dict = {}

    try:
        precision_k_metrics = p_at_tb_k(test_pred, ytest)

        for metric in precision_k_metrics:
            k, p_at_topk, kd_at_topk = metric[1], metric[2], metric[4]
            metrics_dict[f'p_at_top{k}'] = p_at_topk
            metrics_dict[f'kd_at_top{k}'] = kd_at_topk

        best_k_metrics = minmax_n_at_k(test_pred, ytest)

        for metric in best_k_metrics:
            k, min_at_k = metric[1], metric[3]
            metrics_dict[f'br_at_{k}'] = min_at_k

        metrics_dict["mae"] = np.mean(abs(test_pred - ytest))
        metrics_dict["rmse"] = metrics.mean_squared_error(
            ytest, test_pred, squared=False
        )
        metrics_dict["pearson"] = np.abs(np.corrcoef(ytest, test_pred)[1, 0])
        metrics_dict["spearman"] = stats.spearmanr(ytest, test_pred)[0]
        metrics_dict["kendalltau"] = stats.kendalltau(ytest, test_pred)[0]
        metrics_dict["kt_2dec"] = stats.kendalltau(
            ytest, np.round(test_pred, decimals=2)
        )[0]
        metrics_dict["kt_1dec"] = stats.kendalltau(
            ytest, np.round(test_pred, decimals=1)
        )[0]
        for k in [10, 20]:
            top_ytest = np.array(
                [y > sorted(ytest)[max(-len(ytest), -k - 1)] for y in ytest]
            )
            top_test_pred = np.array(
                [
                    y > sorted(test_pred)[max(-len(test_pred), -k - 1)]
                    for y in test_pred
                ]
            )
            metrics_dict["precision_{}".format(k)] = (
                sum(top_ytest & top_test_pred) / k
            )
        metrics_dict["full_ytest"] = ytest.tolist()
        metrics_dict["full_testpred"] = test_pred.tolist()

    except:
        for metric in METRICS:
            metrics_dict[metric] = float("nan")
    if np.isnan(metrics_dict["pearson"]) or not np.isfinite(
        metrics_dict["pearson"]
    ):
        logger.info("Error when computing metrics. ytest and test_pred are:")
        logger.info(ytest)
        logger.info(test_pred)

    return metrics_dict

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AverageMeterGroup:
    """Average meter group for multiple average meters, ported from Naszilla repo."""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = NamedAverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())


class NamedAverageMeter:
    """Computes and stores the average and current value, ported from naszilla repo"""

    def __init__(self, name, fmt=":f"):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Cutout(object):
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
            h, w = img.size(1), img.size(2)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img *= mask
        return img


from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
from fvcore.common.file_io import PathManager
import os


class Checkpointer(fvCheckpointer):
    def load(self, path: str, checkpointables: Optional[List[str]] = None) -> object:
        """
        Load from the given checkpoint. When path points to network file, this
        function has to be called on all ranks.
        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            path = PathManager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if (
            incompatible is not None
        ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:  # pyre-ignore
                self.logger.info("Loading {} from {}".format(key, path))
                obj = self.checkpointables[key]
                try:
                    obj.load_state_dict(checkpoint.pop(key))  # pyre-ignore
                except:
                    print("exception loading")

        # return any further checkpoint data
        return checkpoint

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)
  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].contiguous().view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res,correct

# def create_exp_dir(path, scripts_to_save=None):
#   if not os.path.exists(path):
#     os.mkdir(path)
#   print('Experiment dir : {}'.format(path))
#
#   if scripts_to_save is not None:
#     os.mkdir(os.path.join(path, 'configs'))
#     for script in scripts_to_save:
#       dst_file = os.path.join(path, 'configs', os.path.basename(script))
#       shutil.copyfile(script, dst_file)

def drop_path(x, drop_prob=0):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    if list(x.size())[0]==4:
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
  return x


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
        Example:
            policy = CIFAR10Policy()
            transformed = policy(image)
        Example as a PyTorch Transform:
            transform = transforms.Compose([
                transforms.Resize(256),
                CIFAR10Policy(),
                transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SubPolicy(object):

    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10}

        # from https://stackoverflow.com/questions/5252170/specify-image
        # -filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (128,) * 4), rot) \
                .convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * \
                 random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * \
                 img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: \
            #     img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: \
                ImageEnhance.Color(img).enhance(
                    1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: \
                ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: \
                ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: \
                ImageEnhance.Contrast(img).enhance(
                    1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: \
                ImageEnhance.Sharpness(img).enhance(
                    1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: \
                ImageEnhance.Brightness(img).enhance(
                    1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img
