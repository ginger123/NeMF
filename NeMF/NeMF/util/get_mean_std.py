# Computes the per-channel mean and std of the training images, as the network receives them
# (after stokes_to_real and any other dataset transforms), for data.mean / data.std in the config.
#
# Usage, from the NeMF/NeMF directory:
#   python -m NeMF.util.get_mean_std [config=configs/microphysics_train.yaml] [data.dataset_name=...] [num_workers=4]
# Any other key=value argument overrides the config, e.g. data.n_training=500.
# example:
# python -m NeMF.util.get_mean_std data.dataset_name=BOMEX_500CCN_10cams_20m_polarization_pyshdom num_workers=8

import numpy as np
import torch
from omegaconf import OmegaConf
from NeMF.microphysics_dataset import get_cloud_microphysics_datasets, trivial_collate


def compute_mean_std(dataset, num_workers=4):
    """
    Accumulates per-channel sums over every camera and pixel of the dataset.
    :param dataset MicrophysicsCloudDataset built with mean=0, std=1 so images are not normalized
    :return (mean, std) numpy arrays of shape (channels,)
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=trivial_collate)
    total = None
    total_sq = None

    count = 0
    for i, batch in enumerate(loader):
        images = np.asarray(batch[0][0], dtype=np.float64)  # (n_cam, C, H, W)
        # initialize only once size is known
        if total is None:
            total = np.zeros(images.shape[1])
            total_sq = np.zeros(images.shape[1])
        total += images.sum(axis=(0, 2, 3))
        total_sq += (images ** 2).sum(axis=(0, 2, 3))
        count += images.shape[0] * images.shape[2] * images.shape[3]
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(loader)}")
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)
    return mean, std


def main():
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(OmegaConf.load(cli.pop("config", "configs/microphysics_train.yaml")), cli)
    num_workers = cfg.pop("num_workers", 4)

    # identity normalization, so the statistics are of the raw (transformed) images
    cfg.data.mean = 0.0
    cfg.data.std = 1.0
    train_dataset, _, _ = get_cloud_microphysics_datasets(cfg)
    assert len(train_dataset) > 0, f"no training clouds found for {cfg.data.dataset_name} (data_root is hardcoded in microphysics_dataset.py)"

    mean, std = compute_mean_std(train_dataset, num_workers)
    print(f"Statistics over {len(train_dataset)} training clouds of {cfg.data.dataset_name}:")
    print(f"mean: {mean.tolist()}")
    print(f"std: {std.tolist()}")


if __name__ == "__main__":
    main()
