from helpers import gridsample_residual
import torch

def shift_by_int(img, x_shift, y_shift, is_res=False):
    if is_res:
        img = img.permute(0, 3, 1, 2)

    x_shifted = torch.zeros_like(img)
    if x_shift > 0:
        x_shifted[..., x_shift:, :]  = img[..., :-x_shift, :]
    elif x_shift < 0:
        x_shifted[..., :x_shift, :]  = img[..., -x_shift:, :]
    else:
        x_shifted = img.clone()

    result = torch.zeros_like(img)
    if y_shift > 0:
        result[..., y_shift:]  = x_shifted[..., :-y_shift]
    elif y_shift < 0:
        result[..., :y_shift]  = x_shifted[..., -y_shift:]
    else:
        result = x_shifted.clone()

    if is_res:
        result = result.permute(0, 2, 3, 1)

    return result


def res_warp_res(res_a, res_b, is_pix_res=True):
    if is_pix_res:
        res_b = 2 * res_b / (res_b.shape[-2])

    if len(res_a.shape) == 4:
        result = gridsample_residual(
                        res_a.permute(0, 3, 1, 2),
                        res_b,
                        padding_mode='border').permute(0, 2, 3, 1)
    elif len(res_a.shape) == 3:
        result = gridsample_residual(
                        res_a.permute(2, 0, 1).unsqueeze(0),
                        res_b.unsqueeze(0),
                        padding_mode='border')[0].permute(1, 2, 0)
    else:
        raise Exception("Residual warping requires BxHxWx2 or HxWx2 format.")

    return result


def res_warp_img(img, res_in, is_pix_res=True, padding_mode='zeros'):

    if is_pix_res:
        res = 2 * res_in / (img.shape[-1])
    else:
        res = res_in

    if len(img.shape) == 4:
        result = gridsample_residual(img, res, padding_mode=padding_mode)
    elif len(img.shape) == 3:
        if len(res.shape) == 3:
            result = gridsample_residual(img.unsqueeze(0),
                                         res.unsqueeze(0), padding_mode=padding_mode)[0]
        else:
            img = img.unsqueeze(1)
            result = gridsample_residual(img,
                                         res, padding_mode=padding_mode).squeeze(1)
    elif len(img.shape) == 2:
        result = gridsample_residual(img.unsqueeze(0).unsqueeze(0),
                                     res.unsqueeze(0),
                                     padding_mode=padding_mode)[0, 0]
    else:
        raise Exception("Image warping requires BxCxHxW or CxHxW format." +
                        "Recieved dimensions: {}".format(len(img.shape)))

    return result


def combine_residuals(a, b, is_pix_res=True):
    return b + res_warp_res(a, b, is_pix_res=is_pix_res)

def upsample_residuals(residuals, factor=2.0):
    original_dim = len(residuals.shape)
    while len(residuals.shape) < 4:
        residuals = residuals.unsqueeze(0)
    res_perm = residuals.permute(0, 3, 1, 2)
    #result = upsampler(residuals.permute(
    #                                 0, 3, 1, 2)).permute(0, 2, 3, 1)
    result = torch.nn.functional.interpolate(res_perm, scale_factor=factor, mode='bicubic').permute(0, 2, 3, 1)
    result *= factor
    while len(result.shape) > original_dim:
        result = result.squeeze(0)
    return result

def downsample_residuals(residuals):
    original_dim = len(residuals.shape)
    while len(residuals.shape) < 4:
        residuals = residuals.unsqueeze(0)
    result = torch.nn.functional.avg_pool2d(residuals.permute(
                                     0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    result /= 2
    while len(result.shape) > original_dim:
        result = result.squeeze(0)
    return result
