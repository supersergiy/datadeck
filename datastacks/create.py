import numpy as np
import shelve
import sys
import json
import os
import time
import six
import copy

from os.path import expanduser

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.parallel import data_parallel
from torchvision import transforms

import augment
from stack_dataset import compile_dataset
from augment import set_up_transformation, apply_transform, generate_transform
import argparse

from pdb import set_trace as st

def create_trainval_data_loaders(augdata_params, input_mip):
    result = {}
    result['loaders'] = {}
    result['transform_def'] = {"loss": {}, "run": {}}

    dsets = {"train": None, "val": None}
    dset_mips = {"train": None, "val": None}
    aug_params = {"train": {}, "val": {}}

    if "dataset_params" in augdata_params.keys():
        #legacy TODO: remove
        raise Exception("LEGACY DATASET DROP SUPPORT")
    else:
        augdset_names = {}
        augdset_names['train'] = augdata_params['train_dset_name']
        augdset_names['val']   = augdata_params['val_dset_name']

        for augdset_type in augdset_names.keys():
            augdset_name = augdset_names[augdset_type]
            augdset_param = augdata_params[augdset_name]

            dset_name = augdset_param['dataset_params']['dataset_name']
            dset_param = augdset_param['dataset_params'][dset_name]

            dsets[augdset_type] = create_dataset(dset_param)
            dset_mips[augdset_type] = dset_param['dataset_mip']
            aug_params[augdset_type] = augdset_param['augment_params']
            print (aug_params[augdset_type])
            print (augdset_type)

    for dset_type in dsets:
        set_up_transformation(aug_params[dset_type],
                        dsets[dset_type], dset_mips[dset_type], input_mip)
        result['loaders'][dset_type] = DataLoader(
            dsets[dset_type], batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False
        )

    return result

def create_trainval_dataset(dataset_params):
    if 'train_dset_name' in dataset_params and \
            'val_dset_name' in dataset_params:
        #make sure params support named dataset
        assert 'train_dataset_list' not in dataset_params
        train_dset_name = dataset_params['train_dset_name']
        val_dset_name = dataset_params['val_dset_name']
        datasets, loaders = create_dataset(dataset_params,
                dataset_list=[train_dset_name, val_dset_name])
        result = []
        result += [datasets[train_dset_name], datasets[val_dset_name]]
        result += [loaders[train_dset_name], loaders[val_dset_name]]
        return datasets[train_dset_name], data
    else:
        raise Exception("LEGACY DATASET DROP SUPPORT")

def create_dataset(dataset_param):
    misc = dataset_param['misc']
    dataset_list = dataset_param['dataset_list']
    dataset = compile_dataset(dataset_list, misc=misc)

    return dataset


def create_datasets(dataset_params, dataset_list=None):
    if 'train_dataset_list' in dataset_params:
        raise Exception("LEGACY DATASET DROP SUPPORT")
    else:
        datasets = {}
        loaders = {}

        if dataset_list is None:
            dataset_list = list(dataset_params.keys())

        for dataset_name in dataset_list:
            dataset_param = dataset_params[dataset_name]
            misc = dataset_param['misc']
            dataset_list = dataset_param['dataset_list']
            dataset = compile_dataset(dataset_list, misc=misc)
            datasets[dataset_name] = dataset

        return datasets


def unpack_path_specs(d, working_folder):
    if isinstance(d, dict):
        if 'path' in d:
            full_path = os.path.expanduser(d['path'])
            old_d = d
            with open(full_path, 'r') as f:
                d = unpack_path_specs(json.load(f), working_folder)
            for k in old_d:
                if k != 'path':
                    d[k] = old_d[k]
        else:
            for k in d.keys():
                d[k] = unpack_path_specs(d[k], working_folder)
    return d
