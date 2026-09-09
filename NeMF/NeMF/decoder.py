# This file contains the code for the decoder in NeMF framework.
# It is based on VIP-CT source code ('https://github.com/ronenroi/VIPCT') by Roi Ronen
# Copyright (c) Roi Ronen et al.
# All rights reserved.
#
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


import torch
from torch import nn
from .mlp_function import MLPWithInputSkips2


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)

class Decoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
            self,
            type,
            average_cams,
            feature_flatten,
            latent_size,
            use_neighbours,
            device,
    ):
        """
        :param type Decoder network type.
        """
        super().__init__()
        self.average_cams = average_cams
        self.feature_flatten = feature_flatten
        out_size = 27 if use_neighbours else 1
        self.type = type
        self.device = device

        if type == 'microphysics_1head_with_mask':

            self.decoder = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),

            ),
            torch.nn.Linear(512, 4*out_size),)

        elif type == 'microphysics_3heads_3out':

            self.decoder0 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),)
            self.decoder1 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),)
            self.decoder2 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),)

        elif type == 'microphysics_3heads_with_mask':

            self.decoder0 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                8,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                2048,  # self.harmonic_embedding.embedding_dim_xyz,
                512,
                input_skips=(5,),
                ),
                torch.nn.Linear(512, 2*out_size),)
            self.decoder1 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),)
            self.decoder2 = nn.Sequential(
                torch.nn.Linear(latent_size, 2048),
                torch.nn.ReLU(True),
                MLPWithInputSkips2(
                    8,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    2048,  # self.harmonic_embedding.embedding_dim_xyz,
                    512,
                    input_skips=(5,),
                ),
                torch.nn.Linear(512, out_size),)
        else:
            NotImplementedError()



    def forward(self, x):
        if self.average_cams:
            x = torch.mean(x,1)
        if self.feature_flatten:
            x = x.reshape(x.shape[0],-1)
        if '3heads_with_mask' in self.type:
            dec0_out = self.decoder0(x)
            return torch.stack([dec0_out[:,0], torch.sigmoid(dec0_out[:,1]), self.decoder1(x).squeeze(-1), self.decoder2(x).squeeze(-1)], dim=-1)
        elif '3heads_3out' in self.type:
            return torch.stack([self.decoder0(x),self.decoder1(x),self.decoder2(x)],dim=-1).squeeze()
        elif '1head_with_mask' in self.type:
            dec_out = self.decoder(x)
            return torch.stack([dec_out[:, 0], torch.sigmoid(dec_out[:, 1]), dec_out[:, 2], dec_out[:, 3]],dim=-1)
        else:
            return self.decoder(x)

    @classmethod
    def from_cfg(cls, cfg, latent_size, use_neighbours=False):
        if hasattr(cfg.decoder, 'out_factors'):
            out_factors = cfg.decoder.out_factors
        else:
            out_factors = None
        return cls(
            type = cfg.decoder.name,
            average_cams=cfg.decoder.average_cams,
            feature_flatten=cfg.decoder.feature_flatten,
            latent_size = latent_size,
            use_neighbours = use_neighbours,
            device=cfg.gpu,
        )

