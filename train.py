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
        #legacy
        raise Exception("Unimplemented")
        if 'run_transform_def' in augdata_params:
            result['transform_def']['run'] = \
                    augdata_params['run_transform_def']
        if 'loss_transform_def' in augdata_params:
            result['transform_def']['loss'] = \
                    augdata_params['loss_transform_def']
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
        #make sure it's legacy dataset
        assert 'train_dataset_list' in dataset_params
        return create_dataset(dataset_params)

def create_dataset(dataset_param):
    misc = dataset_param['misc']
    dataset_list = dataset_param['dataset_list']
    dataset = compile_dataset(dataset_list, misc=misc)

    return dataset


def create_datasets(dataset_params, dataset_list=None):
    if 'train_dataset_list' in dataset_params:
        return create_dataset_legacy(dataset_params)
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


def create_dataset_legacy(dataset_params):
    print ("Using legacy dataset format")
    train_misc = dataset_params['train_misc']
    if 'train_crop' in dataset_params:
        train_crop = dataset_params['train_crop']
        train_misc['crop'] = train_crop

    train_dataset_list = dataset_params['train_dataset_list']
    train_dataset = compile_dataset(train_dataset_list, misc=train_misc)

    val_misc = dataset_params['val_misc']
    if 'val_crop' in dataset_params:
        val_crop = dataset_params['val_crop']
        val_misc['crop'] = val_crop

    val_dataset_list = dataset_params['val_dataset_list']
    val_dataset = compile_dataset(val_dataset_list, misc=val_misc)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False
    )
    return train_dataset, val_dataset, train_loader, val_loader

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

def parse_params(args):
    params = {}
    if hasattr(args, 'params_path') and args.params_path is not None:
        params_path = os.path.expanduser(args.params_path)
        params_dir = os.path.dirname(params_path)

        with open(params_path, 'r') as params_file:
            params = json.load(params_file)
        params['name'] = os.path.basename(params_path).split('.')[0]
        params['params_folder'] = params_dir

    for p in ['model_params', 'train_params', 'data_params']:
        if 'params_folder' in params:
            working_folder = params['params_folder']
        else:
            working_folder = None

        if hasattr(args, p) and getattr(args, p) is not None:
            p_path = os.path.expanduser(getattr(args, p))
            if p in params:
                print ("Overriding {} from experiment params with {}".format(p, p_path))

            if p != 'model_params':
                with open(p_path, 'r') as p_file:
                    params[p] = json.load(p_file)
                working_folder = os.path.dirname(p_path)
            else:
                params[p] = {'spec_path': p_path}

        params[p] = unpack_path_specs(params[p], working_folder)

    if hasattr(args, 'name') and getattr(args, 'name') is not None:
        print ("Overriding {} from experiment params with {}".format('name', args.name))
        params['name'] = args.name

    return params
