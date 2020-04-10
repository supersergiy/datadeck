from torch.utils.data import Dataset, ConcatDataset
import copy
import os
import warnings
import torch
import numpy as np
import skimage
from augment import apply_transform
import augment
from scipy import ndimage
from torch import nn
import random
import sys
from pdb import set_trace as st
import six
from residuals import upsample_residuals

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

mask_types = ['edges', 'defects', 'plastic']


def compile_dataset(dataset_specs, transform=None, misc={}):
    datasets = []
    for spec in dataset_specs:
        d = h5_to_dataset_list(spec, transform=transform, misc=misc)
        datasets.extend(d)
    return ConcatDataset(datasets)


def h5_to_dataset_list(spec, transform=None, misc={}):
    """Create a list of StackDatasets from H5 file created by gen_stack.py
    """
    stacks = []
    img_h5 = h5py.File(os.path.expanduser(spec['data']['h5']), 'r')
    img_layer_name = 'main'
    if 'layer' in spec['data']:
        img_layer_name = spec['data']['layer']
    img_dset = img_h5[img_layer_name]

    img_mip = None
    if 'mip' in spec['data']:
        img_mip = spec['data']['mip']


    mask_dsets = {}
    mask_mips = {}
    if 'masks' in spec:
        for mask_type in mask_types:
            if mask_type in spec['masks']:
                #print ("{} mask detected!".format(mask_type))
                mask_h5 = h5py.File(os.path.expanduser(spec['masks'][mask_type]['h5']), 'r')

                mask_layer_name = 'main'
                if 'layer' in spec['masks'][mask_type]:
                    mask_layer_name = spec['masks'][mask_type]['layer']
                mask_dsets[mask_type] = mask_h5[mask_layer_name]
                if 'mip' in spec['masks'][mask_type]:
                    mask_mips[mask_type] = spec['masks'][mask_type]['mip']

    misc_dsets = {}
    misc_mips = {}
    if 'misc_data' in spec:
        for misc_type, misc_path in six.iteritems(spec['misc_data']):
            misc_h5 = h5py.File(os.path.expanduser(spec['misc_data'][misc_type]['h5']), 'r')

            misc_layer_name = 'main'
            if 'layer' in spec['misc_data'][misc_type]:
                misc_layer_name = spec['misc_data'][misc_type]['layer']
            misc_dsets[misc_type] = misc_h5[misc_layer_name]
            if 'mip' in spec['misc_data'][misc_type]:
                misc_mips[misc_type] = spec['misc_data'][misc_type]['mip']


    stacks.append(StackDataset(full_dataset=img_dset, dataset_mip=img_mip,
                               mask_mips=mask_mips, masks=mask_dsets,
                               misc_mips=misc_mips, misc_dsets=misc_dsets,
                               transform=transform, misc=misc))
    return stacks


class StackDataset(Dataset):
    """Deliver consecutive image pairs from 3D image stack
    Args:
        stack (4D ndarray): 1xZxHxW image array
    """

    def is_standard_dataset(self):
        return ("dataset_type" not in self.misc) or (self.misc["dataset_type"] == "standard")

    def is_pairs_dataset(self):
        return self.misc["dataset_type"] == "pairs"

    def is_serial_dataset(self):
        return self.misc["dataset_type"] == "serial"

    def is_supervised_dataset(self):
        return ("is_supervised" not in self.misc) or self.misc["is_supervised"]

    def get_lowest_batch(self):
        lowest = 0
        if 'batch_constraint' in self.misc:
            lowest = self.misc['batch_constraint'][0]
        return lowest

    def get_highest_batch(self):
        highest = self.shape[0]

        if 'batch_constraint' in self.misc:
            highest = min(self.misc['batch_constraint'][1], highest)
        return highest


    def get_lowest_section(self):
        lowest = 0
        if 'section_constraint' in self.misc:
            lowest = self.misc['section_constraint'][0]
        return lowest

    def get_highest_section(self):
        if self.is_pairs_dataset():
            highest = self.shape[0]
        else:
            highest = self.shape[1]

        if 'section_constraint' in self.misc:
            highest = min(self.misc['section_constraint'][1], highest)
        return highest

    def __init__(self, full_dataset, transform=None,
                 dataset_mip=None, misc={},
                 masks={}, mask_mips={},
                 misc_dsets={}, misc_mips={}):


        self.misc = misc
        self.full_dataset = full_dataset

        # constraining shape helps save memory and loading time. better than crop
        self.shape = list(self.full_dataset.shape)
        if 'shape_constraint' in self.misc:
            self.shape[-2:] = self.misc['shape_constraint']

        if 'xy_offset' in self.misc:
            self.xy_offset = self.misc['xy_offset']
            self.shape[-2] -= self.misc['xy_offset'][0]
            self.shape[-1] -= self.misc['xy_offset'][1]
        else:
            self.xy_offset = [0, 0]


        if self.is_standard_dataset():
            self.N = (self.get_highest_batch() - self.get_lowest_batch()) * \
                    (self.get_highest_section() - self.get_lowest_section())
            if not self.is_supervised_dataset():
                self.N -= 1
        elif self.is_pairs_dataset():
            self.N = self.get_highest_section() - self.get_lowest_section()
            if self.is_supervised_dataset():
                raise Exception("Pair dataset is not impolemented for supervised")
        elif self.is_serial_dataset():
            self.N = self.full_dataset.shape[1] * self.full_dataset.shape[0]
            self.N = self.shape[0] * (self.get_highest_section() - self.get_lowest_section())
            if self.is_supervised_dataset():
                raise Exception("Serial dataset is not impolemented for supervised")
        self.transform = transform
        self.masks = masks
        self.mask_mips = mask_mips
        self.misc_dsets = misc_dsets
        self.misc_mips = misc_mips
        self.dataset_mip = dataset_mip
        self.upsampler = torch.nn.functional.upsample

    def set_transform(self, transform):
        self.transform = transform

    def get_ij_index(self, sample_id):
        if self.is_pairs_dataset():
            divider = 1
            i_offset = self.get_lowest_section()
            j_offset = 0
        else:
            divider = self.get_highest_section() - self.get_lowest_section()
            i_offset = self.get_lowest_batch()
            j_offset = self.get_lowest_section()

        i = sample_id // divider + i_offset
        j = sample_id % divider + j_offset

        return [i, j]

    def __len__(self):
        return self.N

    def preprocess(self, bundle):
        if 'preprocess' in self.misc:
            if 'div' in self.misc['preprocess']:
                bundle['src'] /= self.misc['preprocess']['div']
                bundle['tgt'] /= self.misc['preprocess']['div']
            if 'sub' in self.misc['preprocess']:
                bundle['src'] -= self.misc['preprocess']['sub']
                bundle['tgt'] -= self.misc['preprocess']['sub']
        return bundle


    def burn_masks(self, bundle):
        if 'burn_masks' in self.misc:
            for k, v in six.iteritems(self.misc['burn_masks']):
                for n in ['src', 'tgt']:
                    mask_name = '{}_{}'.format(n, k)
                    if mask_name in bundle:
                        mask = bundle[mask_name]#.cpu().numpy()
                        bundle[n][mask > 0.05] = v

        return bundle

    def __getitem__(self, sample_id):
        bundle = {}
        '''for k in ['src', 'tgt', 'src_plastic', 'src_edges', 'src_defects',
                'tgt_plastic', 'tgt_defects']:
                bundle[k] = torch.zeros([2*1024, 2*1024], device="cuda:0")
                #print (bundle[k].shape)
                #import pdb; pdb.set_trace()
        #return bundle'''
        src_ij = self.get_ij_index(sample_id)

        if 'crop' in self.misc:
            shift_granularity = 8
            if self.misc['crop']['type'] == 'random':
                if 'seed' in self.misc['crop']:
                   np.random.seed(self.misc['crop']['seed'])
                x_bot, x_top, y_bot, y_top = augment.get_random_crop_coords(self.shape[2:4], self.misc['crop']['shape'], coord_granularity=shift_granularity)
                np.random.seed()
            elif self.misc['crop']['type'] == 'center':
                x_bot, x_top, y_bot, y_top = augment.get_center_crop_coords(self.shape[2:4], self.misc['crop']['shape'], coord_granularity=shift_granularity)
        else:
            x_bot = y_bot = 0
            x_top = self.shape[-2]
            y_top = self.shape[-1]

        x_bot += self.xy_offset[0]
        x_top += self.xy_offset[0]
        y_bot += self.xy_offset[1]
        y_top += self.xy_offset[1]

        src = self.full_dataset[src_ij[0], src_ij[1], x_bot:x_top, y_bot:y_top]
        if self.is_supervised_dataset():
            tgt = copy.copy(src)
            tgt_ij = copy(copy)
            res = torch.zeros((src.shape[0], src.shape[1], 2), device='cuda')
            bundle['res'] = res
        elif self.is_standard_dataset():
            tgt_ij = copy.copy(src_ij)
            if 'neighbor_range' in self.misc:
                neighbor_range = self.misc['neighbor_range']
                min_section = max(self.get_lowest_section(), src_ij[1] - neighbor_range)
                max_section = min(self.get_highest_section(), src_ij[1] + neighbor_range)
                tgt_id_choices = list(range(min_section, max_section + 1))
                coin = np.random.uniform()
                if coin < 0.9:
                    tgt_id_choices.remove(src_ij[1])

                tgt_ij[1] = random.choice(tgt_id_choices)
                offset = tgt_ij[1] - src_ij[1]
                tgt_id_choices.remove(src_ij[1])
                print ('offset: {}'.format(offset))
            else:
                tgt_ij[1] = src_ij[1] + 1

            tgt = self.full_dataset[tgt_ij[0], tgt_ij[1], x_bot:x_top, y_bot:y_top]
        elif self.is_pairs_dataset():
            tgt_ij = copy.copy(src_ij)
            tgt_ij[1] = 1
            tgt = self.full_dataset[tgt_ij[0], tgt_ij[1], x_bot:x_top, y_bot:y_top]
        elif self.is_serial_dataset():
            tgt = np.zeros_like(src)
            # no target for serial dataset
        bundle['src'] = src
        bundle['tgt'] = tgt
        masks = []
        for mask_type in mask_types:
            if self.dataset_mip is not None and mask_type in self.mask_mips:
                mask_mip = self.mask_mips[mask_type]

                if self.dataset_mip > mask_mip:
                    raise Exception("Not implemented")
                mask_mip_diff = mask_mip - self.dataset_mip
            else:
                mask_mip_diff = 0

            mask_coord_div = 2**mask_mip_diff
            if x_bot % mask_coord_div != 0 or \
               x_top % mask_coord_div != 0 or \
               y_bot % mask_coord_div != 0 or \
               y_top % mask_coord_div != 0:
                   raise RuntimeError("Bad image size. Mask will not align by {}: {}".format(mask_coord_div,
                                                                                            (x_bot, x_top, y_bot, y_top)))

            m_x_bot = x_bot // mask_coord_div
            m_x_top = x_top // mask_coord_div
            m_y_bot = y_bot // mask_coord_div
            m_y_top = y_top // mask_coord_div

            if mask_type in self.masks:
                src_mask = self.masks[mask_type][src_ij[0], src_ij[1], m_x_bot:m_x_top, m_y_bot:m_y_top]
                if self.is_serial_dataset():
                    tgt_mask = np.zeros_like(src_mask)
                else:
                    tgt_mask = self.masks[mask_type][tgt_ij[0], tgt_ij[1], m_x_bot:m_x_top, m_y_bot:m_y_top]
            else:
                src_mask = np.zeros_like(src)
                tgt_mask = np.zeros_like(tgt)
            src_mask = torch.cuda.FloatTensor(src_mask)
            tgt_mask = torch.cuda.FloatTensor(tgt_mask)
            src_tgt_mask = torch.cat((src_mask.unsqueeze(0), tgt_mask.unsqueeze(0)), 0).unsqueeze(0)
            src_tgt_mask = self.upsampler(src_tgt_mask, scale_factor=2.0**mask_mip_diff,
                                          mode='bilinear').squeeze()

            src_mask = src_tgt_mask[0]
            tgt_mask = src_tgt_mask[1]

            if torch.min(src_mask) < 0:
                src_mask -= torch.min(src_mask)
            if torch.max(src_mask) > 1:
                src_mask /= torch.max(src_mask)
            if torch.min(tgt_mask) < 0:
                tgt_mask -= torch.min(tgt_mask)
            if torch.max(tgt_mask) > 1:
                tgt_mask /= torch.max(tgt_mask)

            bundle["src_{}".format(mask_type)] = src_mask
            bundle["tgt_{}".format(mask_type)] = tgt_mask

        for misc_type in self.misc_dsets.keys():
            if self.dataset_mip is not None and misc_type in self.misc_mips:
                misc_mip = self.misc_mips[misc_type]

                if self.dataset_mip > misc_mip:
                    raise Exception("Not implemented")
                misc_mip_diff = misc_mip - self.dataset_mip
            else:
                misc_mip_diff = 0

            misc_coord_div = 2**misc_mip_diff
            if x_bot % misc_coord_div != 0 or \
               x_top % misc_coord_div != 0 or \
               y_bot % misc_coord_div != 0 or \
               y_top % misc_coord_div != 0:
                   raise RuntimeError("Bad image size. {} will not align by {}: {}".format(misc_coord_dif, misc_coord_div,
                                                                                            (x_bot, x_top, y_bot, y_top)))

            m_x_bot = x_bot // misc_coord_div
            m_x_top = x_top // misc_coord_div
            m_y_bot = y_bot // misc_coord_div
            m_y_top = y_top // misc_coord_div
            misc_data_raw = self.misc_dsets[misc_type][src_ij[0]:src_ij[0]+1, :, m_x_bot:m_x_top, m_y_bot:m_y_top]
            if self.dataset_mip is not None:
                misc_data_raw /= 2**self.dataset_mip
            if 'field' in misc_type:
                misc_data = torch.cuda.FloatTensor(misc_data_raw).permute(1, 2, 0)
                misc_data = upsample_residuals(misc_data, factor=2.0**misc_mip_diff)
                bundle[misc_type] = misc_data
            else:
                misc_data = torch.cuda.FloatTensor(misc_data_raw)
                misc_data = self.upsampler(misc_data,
                        scale_factor=2.0**mask_mip_diff,
                        mode='bilinear')
                bundle['src_{}'.format(misc_type)] = \
                        misc_data[:, src_ij[1]:src_ij[1]+1]
                bundle['tgt_{}'.format(misc_type)] = \
                        misc_data[:, tgt_ij[1]:tgt_ij[1]+1]

        #raw_bundle = ((src, tgt, res), masks)
        proc_bundle = self.preprocess(bundle)
        proc_bundle = apply_transform(proc_bundle, self.transform)
        final_bundle = self.burn_masks(proc_bundle)
        return final_bundle

