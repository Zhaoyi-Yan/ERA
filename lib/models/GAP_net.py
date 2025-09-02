"""resnet implemented in torchvision:
https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
"""
import torch
from torch import Tensor
import torch.nn as nn
from norm_ema_quantizer import NormEMAVectorQuantizer
from typing import Type, Any, Callable, Union, List, Optional

BN_MOMENTUM = 0.01
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




"""
c -> c, c

"""

class Block_fc_enc(nn.Module):
    def __init__(self, in_c, last=False, act_type='relu'):
        super(Block_fc_enc, self).__init__()
        if act_type.lower() == 'relu':
            act_layer = nn.ReLU(True)
        elif act_type.lower() == 'mish':
            act_layer = nn.Mish(True)
        elif act_type.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_type.lower() == 'gelu':
            act_layer = nn.GELU()
        elif act_type.lower() == 'leaky':
            act_layer = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f'Not implemented: {act_layer}')
            
        
        if not last:
            self.block = nn.Sequential(
                nn.Linear(in_c, in_c),
                nn.BatchNorm1d(in_c, momentum=BN_MOMENTUM),
                act_layer,
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(in_c, in_c),
                nn.BatchNorm1d(in_c, momentum=BN_MOMENTUM),
            )
    
    def forward(self, x):
        return self.block(x)

class Block_conv_enc(nn.Module):
    def __init__(self, in_c, last=False, act_type='relu'):
        super(Block_conv_enc, self).__init__()
        if act_type.lower() == 'relu':
            act_layer = nn.ReLU(True)
        elif act_type.lower() == 'mish':
            act_layer = nn.Mish(True)
        elif act_type.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_type.lower() == 'gelu':
            act_layer = nn.GELU()
        elif act_type.lower() == 'leaky':
            act_layer = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f'Not implemented: {act_layer}')
            
        
        if not last:
            self.block = nn.Sequential(
                conv3x3(in_c, in_c),
                nn.BatchNorm2d(in_c, momentum=BN_MOMENTUM),
                act_layer,
            )
        else:
            self.block = nn.Sequential(
                conv3x3(in_c, in_c),
                nn.BatchNorm2d(in_c, momentum=BN_MOMENTUM),
            )
    
    def forward(self, x):
        return self.block(x)

class Block_fc_dec(nn.Module):
    def __init__(self, in_c, last=False, act_type='relu'):
        super(Block_fc_dec, self).__init__()
        if act_type.lower() == 'relu':
            act_layer = nn.ReLU(True)
        elif act_type.lower() == 'mish':
            act_layer = nn.Mish(True)
        elif act_type.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_type.lower() == 'gelu':
            act_layer = nn.GELU()
        elif act_type.lower() == 'leaky':
            act_layer = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f'Not implemented: {act_layer}')
        
        
        if not last:
            self.block = nn.Sequential(
                act_layer,
                nn.Linear(in_c, in_c),
                nn.BatchNorm1d(in_c, momentum=BN_MOMENTUM),
            )
        else:  # for dec, the last contains no norm and no activation
            self.block = nn.Sequential(
                nn.Linear(in_c, in_c),
            )
    
    def forward(self, x, x_enc):
        x = x + x_enc
        return self.block(x)

class GAP_net(nn.Module):
    def __init__(self, n_blocks, in_c, act_type='relu'):
        super(GAP_net, self).__init__()

        blocks = [Block_fc_enc(in_c, act_type=act_type) for _ in range(n_blocks-1)] + [Block_fc_enc(in_c, last=True, act_type=act_type)]
        self.pred_res = nn.Sequential(*blocks)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        out = self.pred_res(x)

        return out

class VQ_layer(nn.Module):
    def __init__(self, embed_dim, out_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        out_dim = embed_dim if out_dim is None else out_dim
        # task layer
        self.layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, out_dim)
        )

    
    def forward(self, x):
        return self.layer(x)
    
# whe n_block =1, it means: enc(1 block) + vq layer + dec (1 block)
class GAP_net_vq(nn.Module):
    def __init__(self, n_blocks, in_c, act_type='relu', args=None):
        super(GAP_net_vq, self).__init__()
        self.quantize = NormEMAVectorQuantizer(
         n_embed=args.n_embed, embedding_dim=args.embed_dim, beta=1.0, kmeans_init=True, decay=0.99,
        )

        blocks_enc = [Block_fc_enc(in_c, act_type=act_type) for _ in range(n_blocks)] + [VQ_layer(in_c)]
        self.encoder = nn.Sequential(*blocks_enc)
        # using block_fc_enc to build decoder
        blocks_dec = [Block_fc_enc(in_c, act_type=act_type) for _ in range(n_blocks)] + [VQ_layer(in_c, out_dim=in_c)]
        self.decoder = nn.Sequential(*blocks_dec)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def encode(self, x):
        to_quantized_features = self.encoder(x)
        to_quantizer_features = to_quantized_features.unsqueeze(-1).unsqueeze(-1) # reshape B*C-> B*C*1*1
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss
    
    def decode(self, quantize, **kwargs):
        # quantize: B*C*1*1
        rec = self.decoder(quantize.squeeze(-1).squeeze(-1))
        return rec
    
    def forward(self, x, **kwargs):

        quantize, embed_ind, emb_loss = self.encode(x)
        xrec = self.decode(quantize)

        return xrec, emb_loss

# Modeling the gap of two features
#
class GAP_net_withconv(nn.Module):
    def __init__(self, n_blocks, in_c, act_type='relu'):
        super(GAP_net_withconv, self).__init__()
        
        # add two convolutional layers

        blocks = [Block_conv_enc(in_c, act_type=act_type) for _ in range(n_blocks-1)] + [Block_conv_enc(in_c, last=True, act_type=act_type)]
        self.pred_res = nn.Sequential(*blocks)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        out = self.pred_res(x)

        return out    


    
class GAP_unet(nn.Module):
    def __init__(self, n_blocks, in_c, act_type='relu'):
        super(GAP_unet, self).__init__()

        # for unet, use leaky relu in enc
        blocks_enc = [Block_fc_enc(in_c, act_type='leaky') for _ in range(n_blocks-1)] + [Block_fc_enc(in_c, last=True, act_type='leaky')]
        blocks_dec = [Block_fc_dec(in_c, act_type='relu') for _ in range(n_blocks-1)] + [Block_fc_dec(in_c, last=True, act_type='relu')]
        
        self.pred_enc_list = nn.ModuleList([nn.Sequential(Block_fc_enc) for Block_fc_enc in blocks_enc])
        self.pred_dec_list = nn.ModuleList([nn.Sequential(Block_fc_dec) for Block_fc_dec in blocks_dec])

        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        enc_xs = []
        for pred_enc in self.pred_enc_list:
            x = pred_enc(x)
            enc_xs.append(x)
        
        for pred_dec, x_enc in zip(self.pred_dec_list, enc_xs):
            x = pred_dec[0](x, x_enc) # pred_dec is the nn.Sequential and pred_dec[0] is the block
            
        return x