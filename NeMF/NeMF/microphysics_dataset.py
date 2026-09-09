# This file contains the code for synthetic cloud microphysics dataset loaders for NeMF.
# It is based on PyTorch3D source code ('https://github.com/facebookresearch/pytorch3d') by FAIR
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# It is also based on VIP-CT source code ('https://github.com/ronenroi/VIPCT') by Roi Ronen
# Copyright (c) Roi Ronen et al.
# All rights reserved.

# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file:
# Inbal Kom Betzer, Roi Ronen, Vadim Holodovsky, Yoav. Y. Schechner and Ilan Koren, 
# "NeMF: Neural Microphysics Fields",
# TBD TPAMI 2024.
#
# Copyright (c) Inbal Kom Betzer. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# author. The author is not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.



import os, glob
from typing import Tuple
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import socket
import random
import scipy.io as sio


ALL_DATASETS = ("BOMEX_polarization_pyshdom_varying_M","CASS_10cams_20m_polarization_pyshdom",
                "BOMEX_500CCN_10cams_20m_polarization_pyshdom","BOMEX_polarization_pyshdom_varsun", "BOMEX_polarization_at3d_sunsync_ida",
                "BOMEX_polarization_pyshdom_varsunsync", "BOMEX_polarization_pyshdom_sunsync_ida", "BOMEX_500CCN_10cams_20m_polarization_pyshdom_ida")


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    batch = np.array(batch, dtype=object).transpose().tolist()
    return batch


def get_cloud_microphysics_datasets(cfg):
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        cfg: Set of parameters for the datasets.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """
    dataset_name = cfg.data.dataset_name

    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"'{dataset_name}' does not refer to a known dataset.")

    if dataset_name == 'BOMEX_500CCN_10cams_20m_polarization_pyshdom':
        data_root = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/'
            #'/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_alittlebit12/'
        data_root_gt = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/'
        #'/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_alittlebit12/'
        #'/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/'
    if dataset_name == 'BOMEX_500CCN_10cams_20m_polarization_pyshdom_ida':
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_alittlebit/'
        # '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/'
        # '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_alittlebit12/'
        data_root_gt = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_alittlebit/'
        # '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_alittlebit12/'
        # '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/'
    elif dataset_name == "CASS_10cams_20m_polarization_pyshdom":
        data_root = '/wdata/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/'
        data_root_gt = '/wdata/inbalkom/NN_Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/'
    elif dataset_name == "BOMEX_polarization_pyshdom_varying_M":
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/var_sats/'
        data_root_gt = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/var_sats/'
        cfg.data.image_size = [116, 116]
    elif dataset_name == "BOMEX_polarization_pyshdom_varsun":
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/var_sun/'
        data_root_gt = '/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/new_clouds/'
        cfg.data.image_size = [116, 116]
    elif dataset_name == "BOMEX_polarization_pyshdom_varsunsync":
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_alittlebit12/'
            #'/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_sunsyncorbit/'
        data_root_gt = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_alittlebit12/'
            #'/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_sunsyncorbit/'
        cfg.data.image_size = [116, 116]
    elif dataset_name == "BOMEX_polarization_pyshdom_sunsync_ida":
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_sunsyncorbit/'
        data_root_gt = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_sunsyncorbit/'
        cfg.data.image_size = [116, 116]
    elif dataset_name == "BOMEX_polarization_pyshdom_sunsync_ida":
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_sunsyncorbit/'
        data_root_gt = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_sunsyncorbit/'
        cfg.data.image_size = [116, 116]
    elif dataset_name == "BOMEX_polarization_at3d_sunsync_ida":
        data_root = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_AT3D/varying_sun_lambertian_surface/'
        data_root_gt = '/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_PYSHDOM/varsun_sunsyncorbit/'
        cfg.data.image_size = [116, 116]


    print(f"Loading dataset {dataset_name}, image size={str(cfg.data.image_size)} ...")
    data_train_paths = [f for f in glob.glob(os.path.join(data_root, "train/cloud*.pkl"))]

    train_len = cfg.data.n_training if cfg.data.n_training>0 else len(data_train_paths)
    data_train_paths = data_train_paths[:train_len]

    n_cam = cfg.data.n_cam
    mean = cfg.data.mean
    std = cfg.data.std
    rand_cam = cfg.data.rand_cam
    train_dataset = MicrophysicsCloudDataset(
        data_train_paths,
        os.path.join(data_root_gt, "train"),
        n_cam=n_cam,
        rand_cam = rand_cam,
        mask_type=cfg.ct_net.mask_type,
        in_channels_num=cfg.backbone.in_channels,
        mean=mean,
        std=std,
        dataset_name = dataset_name,
    )

    val_paths = [f for f in glob.glob(os.path.join(data_root, "validation/cloud*.pkl"))]
    val_len = cfg.data.n_val if cfg.data.n_val > 0 else len(val_paths)
    val_paths = val_paths[:val_len]
    val_dataset = MicrophysicsCloudDataset(val_paths, os.path.join(data_root_gt, "validation"), n_cam=n_cam,
        rand_cam = rand_cam, mask_type=cfg.ct_net.val_mask_type, in_channels_num=cfg.backbone.in_channels, mean=mean, std=std,   dataset_name = dataset_name)

    test_paths = [f for f in glob.glob(os.path.join(data_root, "test/cloud*.pkl"))]
    test_dataset = MicrophysicsCloudDataset(test_paths, os.path.join(data_root_gt, "test"), n_cam=n_cam,
                                          rand_cam=rand_cam, mask_type=cfg.ct_net.val_mask_type, in_channels_num=cfg.backbone.in_channels, mean=mean, std=std,
                                          dataset_name=dataset_name)

    return train_dataset, val_dataset, test_dataset


class MicrophysicsCloudDataset(Dataset):
    def __init__(self, cloud_dir, cloud_gt_dir, n_cam, rand_cam=False, transform=None, target_transform=None, mask_type=None,
                 in_channels_num=3, mean=0, std=1, dataset_name=''):
        self.cloud_dir = cloud_dir
        self.cloud_gt_dir = cloud_gt_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mask_type = mask_type
        self.in_channels_num = in_channels_num
        self.n_cam = n_cam
        self.rand_cam = rand_cam
        self.mean = mean
        self.std = std
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.cloud_dir)

    def __getitem__(self, idx):
        cloud_path = self.cloud_dir[idx]
        image_index = cloud_path.split('cloud_results_')[-1].split('.pkl')[0]
        gt_grid_path = os.path.join(self.cloud_gt_dir, "cloud_results_" + str(image_index) + ".pkl")
        with open(gt_grid_path, 'rb') as f:
            gt_grid_data = pickle.load(f)
        with open(cloud_path, 'rb') as f:
            data = pickle.load(f)
        if 'at3d_sunsync' in self.dataset_name:
            sun_index = torch.randperm(5)[0]
        if 'ida' in self.dataset_name:
            if 'at3d_sunsync' in self.dataset_name:
                images = data['images_scatter'][sun_index, :, :self.in_channels_num, :, :]
            else:
                images = data['scatter_images'][:, :self.in_channels_num]
        else:
            if 'at3d_sunsync' in self.dataset_name:
                images = data['images'][sun_index, :, :self.in_channels_num, :, :]
            else:
                images = data['images'][:, :self.in_channels_num]
        mask = None
        if self.mask_type == 'space_carving':
            if ("varsun" in self.dataset_name):
                mask = gt_grid_data['mask']
            else:
                if 'at3d_sunsync' in self.dataset_name:
                    mask = data['mask'][sun_index, :, :, :]
                else:
                    mask = data['mask']
        elif self.mask_type == 'space_carving_morph':
            if 'at3d_sunsync' in self.dataset_name:
                mask = data['mask_morph'][sun_index, :, :, :]
            else:
                mask = data['mask_morph']
        elif self.mask_type == 'gt_mask':
            mask = (gt_grid_data['lwc_gt'] > 0) * (gt_grid_data['reff_gt'] > 0) * (gt_grid_data['veff_gt'] > 0)
        elif self.mask_type == 'gt_mask_mod':
            mask = gt_grid_data['gtmask_morph']*0.5
            gt_mask = (gt_grid_data['lwc_gt'] > 0) * (gt_grid_data['reff_gt'] > 0) * (gt_grid_data['veff_gt'] > 0)
            mask[gt_mask==1]=1
        # if mask.dtype != 'bool':
        #     mask = mask>0

        cam_i = torch.arange(self.n_cam)
        if 'varying' in self.dataset_name:
            index = torch.randperm(5)[0]
            mask = mask[index] if mask is not None else None
            images = images[index]
            camera_center = data['cameras_pos'][index, cam_i]
            projection_matrix = data['cameras_P'][index, cam_i]
        elif 'at3d_sunsync' in self.dataset_name:
            camera_center = data['cameras_pos'][sun_index, cam_i]
            projection_matrix = data['cameras_P'][sun_index, cam_i]
        else:
            camera_center = data['cameras_pos'][cam_i]
            projection_matrix = data['cameras_P'][cam_i]
        if ("CASS" in self.dataset_name) and ("pyshdom" in self.dataset_name) and ("CASS" in cloud_path):
            images = np.squeeze(images[:,cam_i,:,:,:])
        else:
            images = images[cam_i, :, :, :]
        images -= np.array(self.mean[:self.in_channels_num]).reshape((1,self.in_channels_num,1,1))
        images /= np.array(self.std[:self.in_channels_num]).reshape((1,self.in_channels_num,1,1))

        microphysics = np.array([gt_grid_data['lwc_gt'],gt_grid_data['reff_gt'],gt_grid_data['veff_gt']])

        if ("CASS" in self.dataset_name) and ("pyshdom" in self.dataset_name) and ("CASS" in cloud_path):
            assert (microphysics.shape[1] == 64 and microphysics.shape[2] == 64 and microphysics.shape[3] == 32)
        elif ("BOMEX" in self.dataset_name) and ("BOMEX" in cloud_path):
            assert (microphysics.shape[1] == 32 and microphysics.shape[2] == 32 and (microphysics.shape[3] == 32 or microphysics.shape[3] == 64))

        if hasattr(data, 'image_sizes'):
            image_sizes = data['image_sizes'][cam_i]
        else:
            image_sizes = [image.shape[1:] for image in images]

        grid = gt_grid_data['grid']

        if ("varsun" in self.dataset_name):
            env_params = np.array([[data['sun_zenith'].squeeze(), data['sun_azimuth'].squeeze()]] * images.shape[0])
        elif 'at3d_sunsync' in self.dataset_name:
            env_params = np.array([[data['sun_zenith'][sun_index].squeeze(), data['sun_azimuth'][sun_index].squeeze()]])
        else:
            env_params = None

        return images, microphysics, grid, image_sizes, projection_matrix, camera_center, mask, env_params
