# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer



'''
import util.misc as misc
ss = 3407
seed = ss + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
'''




class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)


        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm


    def forward_features(self, x):
        B = x.shape[0]

        ww = int(x.shape[3]/2)

        rgb = x[:, :, :, :ww]
        hha = x[:, :, :, ww:]

        rgb = self.patch_embed(rgb)
        hha = self.patch_embed2(hha)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        rgb = torch.cat((cls_tokens, rgb), dim=1)
        hha = torch.cat((cls_tokens, hha), dim=1)
        rgb = rgb + self.pos_embed
        hha = hha + self.pos_embed

        #'''
        if (self.training):
            import random
            a = (random.random())
            #print(a)
            if ((0.5 <= a) and (a < 0.75)):
            #if(a<0.5):
                rgb = rgb[: ,0:1 , :]
            elif (a >= 0.75):
            #elif (a >= 0.5):
                hha = hha[:, 0:1, :]

        #'''

        x = torch.cat((rgb, hha), 1)
        x = self.pos_drop(x)


        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            #'''
            if (not self.training):
                index = int(x.size(1) / 2)
                x1 = x[:, 1:index, :]
                x2 = x[:, index + 1:, :]
                x = torch.cat((x1, x2), 1)
            else:
                if (a<0.5):
                #if (a>100):
                    index = int(x.size(1) / 2)
                    x1 = x[:, 1:index, :]
                    x2 = x[:, index + 1:, :]
                    x = torch.cat((x1, x2), 1)
                if((0.5 <= a) and (a < 0.75)):
                #if (a < 0.5):
                    x=x[:,2:,:]
                if(a>=0.75):
                #if (a >= 0.5):
                    x=x[:,1:197,:]
            #'''



            x = x.mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = (x[:, 0] + x[:, 197])/2

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model