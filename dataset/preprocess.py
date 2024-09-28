import numpy as np
import torch
from albumentations import Compose
from dataset.stereo_albumentation import Normalize, ToTensor

# use ImageNet stats for normalization
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

normalization = Compose([Normalize(always_apply=True),
                         ToTensor(always_apply=True)], p=1.0)


def get_occluded(w, disp, mode='left'):
    """
    Compute occluded region on the left image border

    :param w: image width
    :param disp: disparity map
    :return: occlusion mask
    """
    x_coords = np.arange(0, w)[None, :]  # 1xW

    if mode.lower() == 'left':
        shifted_x = x_coords - disp
        occ_mask = shifted_x < 0  # True where occluded
    elif mode.lower() == 'right':
        shifted_x = x_coords + disp
        occ_mask = shifted_x > w
    return occ_mask


def custom_transform(inputs, transformation):
    """
    clean out-of-range disparity values and exclude occluded areas

    :param inputs: a dictionary with images and disparity maps
    :return: a dictionary after transformation
    """

    if transformation:
        inputs = transformation(**inputs)

    w = inputs['disp'].shape[-1]
    # clamp disparity values to be within [0, width]
    inputs['disp'] = np.clip(inputs['disp'], 0, w)

    # compute occlusion for the left image
    left_occl = get_occluded(w, inputs['disp'], 'left')
    inputs['occ_mask'][left_occl] = True
    inputs['occ_mask'] = np.ascontiguousarray(inputs['occ_mask'])

    # compute occlusion for the right image
    try:
        right_occl = get_occluded(w, inputs['disp_right'], 'right')
        inputs['occ_mask_right'][right_occl] = 1
        inputs['occ_mask_right'] = np.ascontiguousarray(inputs['occ_mask_right'])
    except KeyError:
        # print('No disp mask right)
        inputs['occ_mask_right'] = np.zeros_like(left_occl, dtype=np.bool_)
    # clean up disparity map
    inputs.pop('disp_right', None)

    # exclude occluded areas
    inputs['disp'][inputs['occ_mask']] = 0
    inputs['disp'] = np.ascontiguousarray(inputs['disp'], dtype=np.float32)

    # return normalized image
    return normalization(**inputs)
