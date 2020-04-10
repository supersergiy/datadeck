import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

import time
import json
import numpy as np
from numpy import random
from copy import copy, deepcopy
import skimage
import six

from helpers import reverse_dim, open_model, expand_dims, get_brightness_mask
from residuals import res_warp_img, res_warp_res, combine_residuals, \
        downsample_residuals

from pdb import set_trace as st

def generate_transform(transform_def, dataset_mip, output_mip):
    transform = [ToFloatTensor()]
    transform_def = {int(m): t for (m, t) in six.iteritems(transform_def)}

    for mip in range(dataset_mip, output_mip + 1):
        if mip in transform_def or str(mip) in transform_def:
            for ot in transform_def[mip]:
                t = deepcopy(ot)
                trans_type = t['type']
                del t['type']
                if trans_type == 'warp':
                    transform.append(RandomWarp(difficulty=t["difficulty"],
                                                max_disp=t["max_disp"],
                                                min_disp=t["min_disp"],
                                                prob=t["prob"],
                                                randomize_d=t["randomize_d"],
                                        random_epicenters=t["random_epicenters"]))
                elif trans_type == 'preprocess':
                    transform.append(Preprocess(**t))
                elif trans_type == 'preprocessor_net':
                    transform.append(PreprocessorNet(**t))
                elif trans_type == 'sergiynorm':
                    transform.append(SergiyNorm(**t))
                elif trans_type == 'lighten':
                    transform.append(Lighten(detector_net_json=t['detector_net_json'],
                                             mip_diff=t['mip_diff']))
                elif trans_type == 'random_transpose':
                    transform.append(RandomTranspose(prob=t['prob']))
                elif trans_type == 'random_src_tgt_swap':
                    transform.append(RandomSrcTgtSwap(prob=t['prob']))
                elif trans_type == 'random_contrast':
                    transform.append(RandomContrast(prob=t['prob'], min_mult=t['min_mult'], max_mult=t['max_mult']))
                elif trans_type == 'random_brightness':
                    transform.append(RandomBrightness(prob=t['prob'], min_add=t['min_add'], max_add=t['max_add']))
                elif trans_type == 'crop_middle':
                    transform.append(CropMiddle(cropped_shape=t["cropped_shape"]))
                elif trans_type == 'random_crop':
                    transform.append(RandomCrop(**t))
                elif trans_type == 'random_unpair_template':
                    transform.append(RandomUnpairTemplate(**t))
                elif trans_type == 'crop_sides':
                    transform.append(CropSides(x_crop=t["x_crop"],
                                               y_crop=t["y_crop"]))
                elif trans_type == 'grey_box':
                    transform.append(GreyBox(min_size=t["min_size"],
                                             max_size=t["max_size"],
                                             prob=t["prob"]))
                elif trans_type == 'black_box':
                    transform.append(GreyBox(min_size=t["min_size"],
                                             max_size=t["max_size"],
                                             prob=t["prob"]))
                elif trans_type == 'sawtooth_boundary':
                    transform.append(SawtoothBoundary(
                                             x_cut=t["x_cut"],
                                             y_cut=t["y_cut"],
                                             do_src=t["do_src"],
                                             do_tgt=t["do_tgt"],
                                             prob=t["prob"],
                                             fill_value=t["fill_value"]))
                elif trans_type == 'translation':
                    RandomTranslation(max_disp=t["max_disp"], prob=t["prob"])
                elif trans_type == 'downsample':
                    transform.append(Downsample())
                else:
                    raise Exception("Unrecognized transformation: {}".format(trans_type))
        if mip < output_mip:
            transform.append(Downsample())

    transform = torchvision.transforms.Compose(transform)
    return transform

def set_up_transformation(transform_def, dataset, dataset_mip, output_mip):
    transform = generate_transform(transform_def, dataset_mip, output_mip)
    for d in dataset.datasets:
        d.set_transform(transform)

def apply_transform(bundle, transform):
    if transform is not None:
        return transform(bundle)
    else:
        return bundle


class Lighten(object):
    def __init__(self, detector_net_json, mip_diff=0):
        self.detector = create_masker(json.loads(detector_net_json))
        self.mip_diff = mip_diff
        self.pooler = torch.nn.AvgPool2d((2, 2))

    def __call__(self, data_and_masks):
        src_tgt_res, masks_var = data_and_masks
        src, tgt, res = src_tgt_res

        src_in = src.unsqueeze(0).unsqueeze(0)
        tgt_in = tgt.unsqueeze(0).unsqueeze(0)
        for _ in range(self.mip_diff):
            src_in = self.pooler(src_in)
            tgt_in = self.pooler(tgt_in)

        src_light = self.detector(src_in)
        tgt_light = self.detector(tgt_in)

        for _ in range(self.mip_diff):
            src_light = torch.nn.functional.interpolate(src_light, scale_factor=2)
            tgt_light = torch.nn.functional.interpolate(tgt_light, scale_factor=2)
        src_lightened = src + src_light.squeeze()
        tgt_lightened = tgt + tgt_light.squeeze()

        return (src_lightened, tgt_lightened, res), masks_var

class PreprocessorNet(object):
    def __init__(self, model_name, checkpoint_folder):
        self.model = open_model(name=model_name, checkpoint_folder=checkpoint_folder)
        self.upsampler = nn.functional.interpolate

    def __call__(self, bundle):
        with torch.no_grad():
            src_proc = self.model(expand_dims(bundle['src'], 4)).squeeze()
            tgt_proc = self.model(expand_dims(bundle['tgt'], 4)).squeeze()
            bundle['src'] = src_proc
            bundle['tgt'] = tgt_proc
            '''
            src_tgt = torch.cat([bundle['src'].unsqueeze(0), bundle['tgt'].unsqueeze(0)], 0)
            good_feature = self.model.intermediate[2][:, feature].unsqueeze(1)
            good_feature_ups = self.upsampler(good_feature, scale_factor=2)
            bundle['src'] = good_feature_ups[0]
            bundle['tgt'] = good_feature_ups[1]'''
        return bundle

class SergiyNorm(object):
    def __init__(self, filter_plastic=False, filter_black=False, filter_defects=False,
                 bad_fill=None,
                 low_threshold=-10000.485, high_threshold=1000, per_feature=False):
        self.normer = torch.nn.InstanceNorm1d(1)
        self.per_slice = True
        self.per_feature = per_feature

        self.filter_black = filter_black
        self.filter_plastic = filter_plastic
        self.filter_defects = filter_defects

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.bad_fill = bad_fill


    def __call__(self, bundle):
        src, tgt = bundle['src'], bundle['tgt']
        # don't require defect mask to run
        # src_defects = (1 - masks_var[2]) * get_raw_defect_mask(src)
        # tgt_defects = (1 - masks_var[3]) * get_raw_defect_mask(tgt)

        # Tissue is the same for all featuremaps
        if src.dim() == 4:
            src_tissue = torch.ones((src.shape[0], 1, src.shape[2], src.shape[3]), device=src.device)
            tgt_tissue = torch.ones_like(src_tissue, device=src.device)
        else:
            src_tissue = torch.ones_like(src, device=src.device)
            tgt_tissue = torch.ones_like(src, device=src.device)

        if self.filter_plastic:
            src_plastic = (1 - bundle['src_plastic'])
            tgt_plastic = (1 - bundle['tgt_plastic'])
            src_tissue *= src_plastic
            tgt_tissue *= tgt_plastic
        if self.filter_black:
            src_white = get_brightness_mask(src, self.low_threshold, self.high_threshold)
            tgt_white = get_brightness_mask(tgt, self.low_threshold, self.high_threshold)
            src_tissue *= src_white
            tgt_tissue *= tgt_white

        if self.filter_defects:
            src_defects = ((1 - bundle['src_defects']) > 0.95).type(torch.cuda.FloatTensor)
            tgt_defects = ((1 - bundle['tgt_defects']) > 0.95).type(torch.cuda.FloatTensor)
            src_tissue *= src_defects
            tgt_tissue *= tgt_defects

        src_good_mask = src_tissue >= 0.99
        tgt_good_mask = tgt_tissue >= 0.99
        src_bad_mask = src_good_mask == False
        tgt_bad_mask = tgt_good_mask == False

        # set a corner dummy value to true, so that we never try to take
        # an empty slice during convertion to 1d
        src_good_mask[..., 0, 0] = 1
        tgt_good_mask[..., 0, 0] = 1

        if tgt.dim() == 3 and self.per_slice:
            # with batch. don't want to normalize accross images
            for i in range(tgt.shape[0]):
                tgt[i][tgt_good_mask[i]] = \
                        self.normer(tgt[i][tgt_good_mask[i]].unsqueeze(0).unsqueeze(0)).squeeze()
                src[i][src_good_mask[i]] = \
                        self.normer(src[i][src_good_mask[i]].unsqueeze(0).unsqueeze(0)).squeeze()
        elif tgt.dim() == 4 and self.per_feature:
            for i in range(tgt.shape[1]):
                src_slice = src[:, i:i+1]
                tgt_slice = tgt[:, i:i+1]
                tgt_slice[tgt_good_mask] = \
                        self.normer(tgt_slice[tgt_good_mask].unsqueeze(0).unsqueeze(0)).squeeze()
                src_slice[src_good_mask] = \
                        self.normer(src_slice[src_good_mask].unsqueeze(0).unsqueeze(0)).squeeze()

        else:
            tgt[tgt_good_mask] = self.normer(tgt[tgt_good_mask].unsqueeze(0).unsqueeze(0)).squeeze()
            src[src_good_mask] = self.normer(src[src_good_mask].unsqueeze(0).unsqueeze(0)).squeeze()

        if self.bad_fill is None:
            src[src_bad_mask] = torch.min(src)
            tgt[tgt_bad_mask] = torch.min(tgt)
        else:
            src[src_bad_mask] = self.bad_fill
            tgt[tgt_bad_mask] = self.bad_fill
        return bundle

class CropSides(object):
    def __init__(self, x_crop, y_crop):
        self.x_crop = x_crop
        self.y_crop = y_crop

    def __call__(self, data_and_masks):
        src_tgt_res, masks_var = data_and_masks
        src, tgt, res = src_tgt_res

        x_size = src.shape[-2]
        y_size = src.shape[-1]

        x_s = self.x_crop[0]
        x_e = -self.x_crop[1] if self.x_crop[1] > 0 else x_size
        y_s = self.y_crop[0]
        y_e = -self.y_crop[1] if self.y_crop[1] > 0 else y_size

        src_cropped = src[x_s:x_e, y_s:y_e]
        tgt_cropped = tgt[x_s:x_e, y_s:y_e]
        res_cropped = res[x_s:x_e, y_s:y_e]

        cropped_masks = []
        for m in masks_var:
            cropped_masks.append(m[x_s:x_e, y_s:y_e])

        return (src_cropped, tgt_cropped, res_cropped), cropped_masks

class CropMiddle(object):
    def __init__(self, cropped_shape):
        self.cropped_shape = cropped_shape

    def __call__(self, bundle):
        src_tgt_res, masks_var = data_and_masks
        src, tgt, res = src_tgt_res
        original_size = src.shape[-1]

        x_bot, x_top, y_bot, y_top = get_center_crop_coords(src.shape, self.cropped_shape)

        src_cropped = src[x_bot:x_top, y_bot:y_top]
        tgt_cropped = tgt[x_bot:x_top, y_bot:y_top]
        res_cropped = res[x_bot:x_top, y_bot:y_top, :]

        cropped_masks = []
        for m in masks_var:
            cropped_masks.append(m[x_bot:x_top, y_bot:y_top])

        return (src_cropped, tgt_cropped, res_cropped), cropped_masks

class RandomUnpairTemplate(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, bundle):
        coin = np.random.uniform()
        if coin < self.prob:
            bundle['src'] = bundle['src'].transpose(-1, -2)
            bundle['paired'] = False
        else:
            bundle['paired'] = True

        return bundle

class RandomCrop(object):
    def __init__(self, cropped_shape, crop_src=True, crop_tgt=True):
        self.cropped_shape = cropped_shape
        self.crop_src = crop_src
        self.crop_tgt = crop_tgt

    def __call__(self, bundle):
        to_crop_src_keys = [i for i in bundle.keys() if 'src' in i]
        to_crop_tgt_keys = [i for i in bundle.keys() if 'tgt' in i]
        to_crop_src = [bundle[i] for i in to_crop_src_keys]
        to_crop_tgt = [bundle[i] for i in to_crop_tgt_keys]

        if self.crop_src and self.crop_tgt:
            cropped = random_crop(to_crop_src + to_crop_tgt, self.cropped_shape)
            cropped_src = cropped[:len(to_crop_src_keys)]
            cropped_tgt = cropped[len(to_crop_src_keys):]

            for i in range(len(to_crop_src_keys)):
                bundle[to_crop_src_keys[i]]= cropped_src[i]
            for i in range(len(to_crop_tgt_keys)):
                bundle[to_crop_tgt_keys[i]]= cropped_tgt[i]

        elif self.crop_src:
            cropped_src = random_crop(to_crop_src, self.cropped_shape)
            for i in range(len(to_crop_src_keys)):
                bundle[to_crop_src_keys[i]]= cropped_src[i]
        elif self.crop_tgt:
            cropped_tgt = random_crop(to_crop_tgt, self.cropped_shape)
            for i in range(len(to_crop_tgt_keys)):
                bundle[to_crop_tgt_keys[i]]= cropped_tgt[i]

        '''original_size = src.shape[-1]

        x_offset = random.randint(0, original_size - self.cropped_size - 1)
        y_offset = random.randint(0, original_size - self.cropped_size - 1)

        src_cropped = src[x_offset:x_offset + self.cropped_size,
                          y_offset:y_offset + self.cropped_size]
        tgt_cropped = tgt[x_offset:x_offset + self.cropped_size,
                          y_offset:y_offset + self.cropped_size]
        res_cropped = res[x_offset:x_offset + self.cropped_size,
                          y_offset:y_offset + self.cropped_size, :]'''

        return bundle

def get_random_crop_coords(full_shape, cropped_shape, coord_granularity=4):
    assert cropped_shape[0] <= full_shape[0]
    assert cropped_shape[1] <= full_shape[1]
    assert cropped_shape[0] % coord_granularity == 0
    assert cropped_shape[1] % coord_granularity == 0

    x_bot_preshift = np.random.randint(0, full_shape[0] - cropped_shape[0] + 1)
    y_bot_preshift = np.random.randint(0, full_shape[1] - cropped_shape[1] + 1)
    x_bot = x_bot_preshift - (x_bot_preshift % coord_granularity)
    y_bot = y_bot_preshift - (y_bot_preshift % coord_granularity)
    x_top = x_bot + cropped_shape[0]
    y_top = y_bot + cropped_shape[1]

    return x_bot, x_top, y_bot, y_top

def get_center_crop_coords(full_shape, cropped_shape, coord_granularity=4):
    assert cropped_shape[0] <= full_shape[0]
    assert cropped_shape[1] <= full_shape[1]

    assert cropped_shape[0] % coord_granularity == 0
    assert cropped_shape[1] % coord_granularity == 0

    x_bot_preshift = (full_shape[0] - cropped_shape[0]) // 2
    y_bot_preshift = (full_shape[1] - cropped_shape[1]) // 2
    x_bot = x_bot_preshift - (x_bot_preshift % coord_granularity)
    y_bot = y_bot_preshift - (y_bot_preshift % coord_granularity)

    x_top = x_bot + cropped_shape[0]
    y_top = y_bot + cropped_shape[1]

    return x_bot, x_top, y_bot, y_top

def random_crop(img, cropped_shape):
    result = []
    if isinstance(img, list):
        original_shape = img[0].shape[-2:]
        x_bot, x_top, y_bot, y_top = get_random_crop_coords(original_shape, cropped_shape)
        for i in img:
            assert (i.shape[-2] == original_shape[-2])
            assert (i.shape[-1] == original_shape[-1])

            result.append(i[..., x_bot:x_top, y_bot:y_top])
    else:
        original_shape = img.shape
        x_bot, x_top, y_bot, y_top = get_random_crop_coords(original_shape, cropped_shape)

        result.append(img[..., x_bot:x_top, y_bot:y_top])
    return result


class RandomTranspose(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, bundle):
        coin = np.random.uniform()
        if coin < self.prob:
            for k, v in six.iteritems(bundle):
                if k != 'res':
                    bundle[k] = v.transpose(0, 1)

            if 'res' in bundle:
                bundle['res'] = bundle['res'].transpose(0, 1).flip(2)

        return bundle

class RandomSrcTgtSwap(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, bundle):
        result = deepcopy(bundle)
        coin = np.random.uniform()
        if coin < self.prob:
            for k, v in six.iteritems(bundle):
                if 'src' in k:
                    result[k.replace('src', 'tgt')] = bundle[k]
                if 'tgt' in k:
                    result[k.replace('tgt', 'src')] = bundle[k]

        return result

class RandomBrightness(object):
    def __init__(self, prob=0.5, min_add=0.01, max_add=0.1):
        self.prob = prob
        self.min_add = min_add
        self.max_add = max_add

    def __call__(self, data_and_masks):
        src_tgt_res, masks_var = data_and_masks
        src, tgt, res = src_tgt_res

        coin_src = np.random.uniform()
        if coin_src < self.prob:
            cutoff = int(np.random.uniform() * src.shape[1])
            add = np.random.uniform(self.min_add, self.max_add)
            coin = np.random.uniform()
            print ('Brighting')
            if coin < 0.5:
                coin = np.random.uniform()
                if coin < 0.5:
                    src[:, cutoff:, :] += add
                else:
                    src[:, :cutoff, :] += add
            else:
                coin = np.random.uniform()
                if coin < 0.5:
                    src[:, :, cutoff:] += add
                else:
                    src[:, :, :cutoff] += add

        coin_tgt = np.random.uniform()
        if coin_tgt < self.prob:
            cutoff = int(np.random.uniform() + tgt.shape[1])
            add = np.random.uniform(self.min_add, self.max_add)
            coin = np.random.uniform()
            if coin < 0.5:
                coin = np.random.uniform()
                if coin < 0.5:
                    tgt[:, cutoff:, :] += add
                else:
                    tgt[:, :cutoff, :] += add
            else:
                coin = np.random.uniform()
                if coin < 0.5:
                    tgt[:, :, cutoff:] += add
                else:
                    tgt[:, :, :cutoff] += add

        return ((src, tgt, res), masks_var)


class RandomContrast(object):
    def __init__(self, prob=0.5, min_mult=1.01, max_mult=1.6):
        self.prob = prob
        self.min_mult = min_mult
        self.max_mult = max_mult

    def __call__(self, data_and_masks):
        src_tgt_res, masks_var = data_and_masks
        src, tgt, res = src_tgt_res

        coin_src = np.random.uniform()
        if coin_src < self.prob:
            cutoff = int(np.random.uniform() * src.shape[1])
            mult = np.random.uniform(self.min_mult, self.max_mult)
            coin = np.random.uniform()
            if coin < 0.5:
                coin = np.random.uniform()
                if coin < 0.5:
                    src[:, cutoff:, :] *= mult
                else:
                    src[:, :cutoff, :] *= mult
            else:
                coin = np.random.uniform()
                if coin < 0.5:
                    src[:, :, cutoff:] *= mult
                else:
                    src[:, :, :cutoff] *= mult

        coin_tgt = np.random.uniform()
        if coin_tgt < self.prob:
            cutoff = int(np.random.uniform() * tgt.shape[1])
            mult = np.random.uniform(self.min_mult, self.max_mult)
            coin = np.random.uniform()
            if coin < 0.5:
                coin = np.random.uniform()
                if coin < 0.5:
                    tgt[:, cutoff:, :] *= mult
                else:
                    tgt[:, :cutoff, :] *= mult
            else:
                coin = np.random.uniform()
                if coin < 0.5:
                    tgt[:, :, cutoff:] *= mult
                else:
                    tgt[:, :, :cutoff] *= mult
        return ((src, tgt, res), masks_var)

class Downsample(object):
    def __init__(self):
        self.pooler = torch.nn.AvgPool2d((2, 2))

    def __call__(self, bundle):

        for k, v in six.iteritems(bundle):

            if 'res' in k or 'field' in k:
                #this might not work when introducing b
                original_size = v.shape[-1]
                assert (original_size % 2 == 0)
                bundle[k] = downsample_residuals(v)
            elif 'src' in k or 'tgt' in k:
                original_size = v.shape[-1]
                assert (original_size % 2 == 0)
                bundle[k] = self.pooler(v.unsqueeze(0)).squeeze(0)

        return bundle


class RandomTranslation(object):
    """ Translate image by disp pixels
    """
    def __init__(self, max_disp=(2**6, 2**6), prob=1.0):
        self.max_disp = max_disp
        self.prob = prob

    def __call__(self, src_tgt_res):
        src, tgt, res = src_tgt_res
        coin = np.random.uniform()
        if coin < self.prob:
            x_disp = np.random.uniform() * self.max_disp[0] - self.max_disp[0] / 2
            tgt, res = x_translation(tgt, x_disp, res)

        coin = np.random.uniform()
        if coin < self.prob:
            y_disp = np.random.uniform() * self.max_disp[1] - self.max_disp[1] / 2
            tgt, res = y_translation(tgt, y_disp, res)

        return src, tgt, res


class Translation(object):
    """ Translate image by disp pixels
    """
    def __init__(self, disp=(2**6, 2**6)):
        self.disp = disp

    def __call__(self, src_tgt_res):
        src, tgt, res = src_tgt_res

        # TODO: combine residuals properly
        tgt, res = x_translation(tgt, self.disp[0], res)
        tgt, res = y_translation(tgt, self.disp[1], res)

        return src, tgt, res


class ToFloatTensor(object):
    """Convert ndarray to FloatTensor
    """
    def __call__(self, bundle):
        for k, v in six.iteritems(bundle):
            bundle[k] = torch.cuda.FloatTensor(v)

        return bundle

class Preprocess(object):
    def __init__(self, div=255.0, sub=0.5):
        self.div = div
        self.sub = sub

    def __call__(self, bundle):
        bundle['src'] = bundle['src'] / self.div - self.sub
        bundle['tgt'] = bundle['tgt'] / self.div - self.sub

        return bundle

class GreyBox(object):
    def __init__(self, max_size=2, min_size=10, prob=0.5,
                 do_src=True, do_tgt=True):
        self.max_size = max_size
        self.min_size = min_size
        self.prob = prob
        self.do_src = do_src
        self.do_tgt = do_tgt

    def __call__(self, src_tgt_res):
        src, tgt, res = src_tgt_res
        src = occlude_box(src, self.min_size, self.max_size,
                          self.prob, fill_proportion=0.7,
                          fill_low=-0.5, fill_high=-0.15)
        tgt = occlude_box(tgt, self.min_size, self.max_size,
                          self.prob, fill_proportion=0.7,
                          fill_low=-0.5, fill_high=-0.15)
        return src, tgt, res


class BlackBox(object):
    def __init__(self, max_size=2, min_size=10, prob=0.5,
                 do_src=True, do_tgt=True):
        self.max_size = max_size
        self.min_size = min_size
        self.prob = prob

    def __call__(self, src_tgt_res):
        src, tgt, res = src_tgt_res
        src = occlude_box(src, self.min_size, self.max_size,
                          self.prob, fill_proportion=1,
                          fill_low=-0.5, fill_high=-0.5)
        tgt = occlude_box(tgt, self.min_size, self.max_size,
                          self.prob, fill_proportion=1,
                          fill_low=-0.5, fill_high=-0.5)
        return src, tgt, res


def occlude_box(img, min_size, max_size, prob, fill_proportion,
                fill_low, fill_high):
    coin = np.random.uniform()
    if coin < prob:
        x_size = random.randint(min_size, max_size)
        y_size = random.randint(min_size, max_size)
        xs = random.randint(0, img.shape[0] - x_size)
        ys = random.randint(0, img.shape[1] - y_size)

        mask         = np.random.uniform(size=(x_size, y_size)) < fill_proportion
        masked_count = np.count_nonzero(mask)
        fill_data    = np.random.uniform(low=fill_low, high=fill_high, size=masked_count)

        target_data = np.array(img[xs:xs+x_size, ys:ys+y_size])
        target_data[mask] = fill_data
        img[xs:xs+x_size, ys:ys+y_size] = torch.FloatTensor(target_data)
    return img


class RandomWarp(object):
    """ Warp With Random Field
    difficulty
        0 == one direction
        1 == 2 directions along each axis
        i == 2^i directions along each axis
    """
    def __init__(self, difficulty=2, max_disp=10, prob=1.0, min_disp=0,
                 randomize_d=False, random_epicenters=True):
        assert difficulty >= 0
        self.randomize_d = randomize_d
        self.difficulty = difficulty
        self.max_disp = max_disp
        self.min_disp = min_disp
        self.prob     = prob
        self.random_epicenters = random_epicenters

    def __call__(self, bundle):
        src = bundle['src']

        if self.randomize_d and self.difficulty > 0:
            curr_diff = random.randint(0, self.difficulty)
        else:
            curr_diff = self.difficulty
        coin = np.random.uniform()
        if coin < self.prob:
            # NOTE image dimention assumed to be power of 2

            granularity = int(np.log2(src.shape[-1]) - curr_diff)
            res_delta = generate_random_residuals(src.squeeze().shape,
                                                  min_disp=self.min_disp,
                                                  max_disp=self.max_disp,
                                                  granularity=granularity,
                                                  random_epicenters=self.random_epicenters)
            for k, v in six.iteritems(bundle):
                if k == 'res':
                    bundle[k] = combine_residuals(v, res_delta,
                                                is_pix_res=True)
                elif 'res' in k:
                    print ("Warp does nothing to {}".format(k))
                elif 'tgt' in k:
                    bundle[k] = res_warp_img(bundle[k], res_delta, is_pix_res=True)

        return bundle



class SawtoothBoundary(object):
    def __init__(self, x_cut, y_cut, do_src, do_tgt, prob,
                 fill_value=-0.5):
        self.x_cut = x_cut
        self.y_cut = y_cut
        self.do_src = do_src
        self.do_tgt = do_tgt
        self.fill_value = fill_value
        self.cut_prob = prob

    def __call__(self, data_and_masks):
        src_tgt_res, masks_var = data_and_masks
        src, tgt, res = src_tgt_res
        cut_functions = [horizontal_sawtooth_left, horizontal_sawtooth_right,
                         vertical_sawtooth_top, vertical_sawtooth_bot]

        for f in cut_functions:
            if self.do_src:
                coin = np.random.uniform()
                if coin < self.cut_prob:
                    src = f(src, self.x_cut, self.y_cut, self.fill_value)
            if self.do_tgt:
                coin = np.random.uniform()
                if coin < self.cut_prob:
                    tgt = f(tgt, self.x_cut, self.y_cut, self.fill_value)

        return ((src, tgt, res), masks_var)


def horizontal_sawtooth_left(img, x_cut, y_cut, fill_value):
    img_width = img.shape[0]
    img_height = img.shape[1]

    cuts = []
    curr_height = random.randint(0, img_height // 6)

    width_i = 0
    while width_i < img_width:
        cut_length = random.randint(1, x_cut)
        if width_i + cut_length > img_width:
            cut_length = img_width - width_i

        delta = random.randint(-y_cut, y_cut)
        curr_height = curr_height + delta
        if curr_height < 0:
            curr_height = 0
        if curr_height >= img_height:
            curr_height = img_height - 1

        cuts.append({"w_start": width_i,
                     "w_end": width_i + cut_length,
                     "height": curr_height})

        width_i += cut_length

    img_out = img
    for c in cuts:
        img_out[c['w_start']:c['w_end'], 0:c['height']] = fill_value

    return img_out


def horizontal_sawtooth_right(img, x_cut, y_cut, fill_value):
    img_width = img.shape[0]
    img_height = img.shape[1]

    cuts = []
    curr_height = (img_height // 6) * 5 + random.randint(0, img_height // 6)

    width_i = 0
    while width_i < img_width:
        cut_length = random.randint(1, x_cut)
        if width_i + cut_length > img_width:
            cut_length = img_width - width_i

        delta = random.randint(-y_cut, y_cut)
        curr_height = curr_height + delta
        if curr_height < 0:
            curr_height = 0
        if curr_height >= img_height:
            curr_height = img_height - 1

        cuts.append({"w_start": width_i,
                     "w_end": width_i + cut_length,
                     "height": curr_height})

        width_i += cut_length

    img_out = img
    for c in cuts:
        img_out[c['w_start']:c['w_end'], c['height']:] = fill_value

    return img_out


def vertical_sawtooth_top(img, x_cut, y_cut, fill_value):
    img_in_trans  = torch.transpose(img, 0, 1)
    img_out_trans = horizontal_sawtooth_left(img_in_trans, x_cut, y_cut, fill_value)
    img_out = torch.transpose(img_out_trans, 0, 1)
    return img_out


def vertical_sawtooth_bot(img, x_cut, y_cut, fill_value):
    img_in_trans  = torch.transpose(img, 0, 1)
    img_out_trans = horizontal_sawtooth_right(img_in_trans, x_cut, y_cut, fill_value)
    img_out = torch.transpose(img_out_trans, 0, 1)
    return img_out


def generate_random_residuals(shape, max_disp, min_disp=0, granularity=9,
                              random_epicenters=True):
    if random_epicenters:
        seed_shape = [i // (2**(granularity - 1)) for i in shape]
        up_shape = [i * 2 for i in shape]
    else:
        seed_shape = [i // (2**(granularity)) for i in shape]
        up_shape = shape

    '''mask_x = np.random.uniform(size=seed_shape) > 0.5
    mask_y = np.random.uniform(size=seed_shape) > 0.5

    seed_x = np.random.normal(size=seed_shape, scale=1.0)
    seed_y = np.random.normal(size=seed_shape, scale=1.0)

    seed_x[mask_x] = np.random.normal(size=[np.count_nonzero(mask_x)],
                                      scale=0.1)
    seed_y[mask_y] = np.random.normal(size=[np.count_nonzero(mask_y)],
                                      scale=0.1)

    seed_x = seed_x * np.random.uniform() * max_disp
    seed_y = seed_y * np.random.uniform() * max_disp'''
    seed_x = np.random.uniform(size=seed_shape, low=min_disp, high=max_disp)
    seed_y = np.random.uniform(size=seed_shape, low=min_disp, high=max_disp)
    up_x = skimage.transform.resize(seed_x, up_shape)
    up_y = skimage.transform.resize(seed_y, up_shape)
    final_x, final_y = random_crop([up_x, up_y], shape)
    result = torch.cuda.FloatTensor(np.stack([final_x, final_y], axis=2))
    return result


def generate_whirpool_residuals(shape, disp=10):
    field = np.array([[[0, -disp], [disp, 0]], [[-disp, 0], [0, disp]]])

    upsampled_x = skimage.transform.resize(field[:, :, 0], shape)
    upsampled_y = skimage.transform.resize(field[:, :, 1], shape)

    return np.stack([upsampled_x, upsampled_y], axis=2)


def x_translation(img, disp, res):
    result        = torch.cuda.zeros(img.size(), device='cuda')
    res_delta = torch.cuda.zeros(res.size(), device='cuda')

    res_delta[:, :, 1] -= disp
    result = res_warp_img(img, res_delta, is_pix_res=True)
    res = combine_residuals(res, res_delta, is_pix_res=True)

    return result, res


def y_translation(img, disp, res):
    result        = torch.cuda.zeros(img.size(), device='cuda')
    res_delta = torch.cuda.zeros(res.size(), device='cuda')

    res_delta[:, :, 0] -= disp
    result = res_warp_img(img, res_delta, is_pix_res=True)

    res = combine_residuals(res, res_delta, is_pix_res=True)

    return result, res
