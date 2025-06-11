"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn

from .common import InstanceNorm, DepthConv, ConvDepthConv, Decoder_DepthConv


class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x

class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding = p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding = p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)


    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out

class ResnetBlock_Adain_DSC(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain_DSC, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [DepthConv(dim, dim, kernel_size=3, padding=p, norm_layer=InstanceNorm)]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [DepthConv(dim, dim, kernel_size=3, padding=p, norm_layer=InstanceNorm)]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)

    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out

#-----------------------------------------------------------

class Generator_Adain_Upsample(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, n_blocks=3, deep=False,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Generator_Adain_Upsample, self).__init__()

        activation = nn.ReLU(True)

        self.deep = deep

        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
                                         norm_layer(64), activation)
        ### downsample
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   norm_layer(128), activation)
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   norm_layer(256), activation)
        self.down3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
                                   norm_layer(128), activation)
        if self.deep:
            self.down4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                                       norm_layer(128), activation)

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [ResnetBlock_Adain(128, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

        ### upsample
        if self.deep:
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128), activation
            )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), activation
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), activation
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), activation
        )

        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0))

    def forward(self, input, dlatents):
        x = input

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)
        bot = []
        bot.append(x)
        features = []
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)
            bot.append(x)

        if self.deep:
            y4 = self.up4(x)
            features.append(y4)
            y3 = self.up3(y4)
            features.append(y3)
        else:
            y3 = self.up3(x)
            features.append(y3)
        y2 = self.up2(y3)
        features.append(y2)
        y1 = self.up1(y2)
        features.append(y1)
        y = self.last_layer(y1)
        # y = (y + 1) / 2

        # return y, bot, features, dlatents
        return y

#-----------------------------------------------------------

class Generator_Adain_Upsample_DSC3(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, n_blocks=3, deep=False,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Generator_Adain_Upsample_DSC3, self).__init__()

        activation = nn.ReLU(True)

        self.deep = deep

        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                         ConvDepthConv(input_nc, 64, 64, kernel_size=7, padding=0, norm_layer=norm_layer, activation=activation))
        ### downsample
        self.down1 = ConvDepthConv(64, 64, 128, 3, 2, norm_layer=norm_layer, activation=activation)
        self.down2 = ConvDepthConv(128, 128, 256, 3, 2, norm_layer=norm_layer, activation=activation)
        self.down3 = ConvDepthConv(256, 128, 128, 3, 2, norm_layer=norm_layer, activation=activation)
        if self.deep:
            self.down4 = ConvDepthConv(128, 128, 128, 3, 2, norm_layer=norm_layer, activation=activation)

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [ResnetBlock_Adain(128, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

        ### upsample
        if self.deep:
            self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                     ConvDepthConv(128, 128, 128, 3, 1, norm_layer=norm_layer, activation=activation))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                 ConvDepthConv(128, 128, 256, 3, 1, norm_layer=norm_layer, activation=activation))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                 ConvDepthConv(256, 128, 128, 3, 1, norm_layer=norm_layer, activation=activation))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                 ConvDepthConv(128, 64, 64, 3, 1, norm_layer=norm_layer, activation=activation))

        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0))

    def forward(self, input, dlatents):
        x = input

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)
        bot = []
        bot.append(x)
        features = []
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)
            bot.append(x)

        if self.deep:
            y4 = self.up4(x)
            features.append(y4)
            y3 = self.up3(y4)
            features.append(y3)
        else:
            y3 = self.up3(x)
            features.append(y3)
        y2 = self.up2(y3)
        features.append(y2)
        y1 = self.up1(y2)
        features.append(y1)
        y = self.last_layer(y1)
        # y = (y + 1) / 2

        # return y, bot, features, dlatents
        return y

#-----------------------------------------------------------

class Generator_Adain_Upsample_DS2(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, n_blocks=3, deep=False,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Generator_Adain_Upsample_DS2, self).__init__()

        activation = nn.ReLU(True)

        self.deep = deep

        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                         DepthConv(input_nc, 64, kernel_size=7, padding=0, norm_layer=norm_layer, activation=activation))
        ### downsample
        self.down1 = DepthConv(64, 128, 3, 2, norm_layer=norm_layer, activation=activation)
        self.down2 = DepthConv(128, 256, 3, 2, norm_layer=norm_layer, activation=activation)
        self.down3 = DepthConv(256, 128, 3, 2, norm_layer=norm_layer, activation=activation)
        if self.deep:
            self.down4 = DepthConv(128, 128, 3, 2, norm_layer=norm_layer, activation=activation)

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [ResnetBlock_Adain(128, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

        ### upsample
        if self.deep:
            self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                     DepthConv(128, 128, 3, 1, norm_layer=norm_layer, activation=activation))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                 DepthConv(128, 256, 3, 1, norm_layer=norm_layer, activation=activation))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                 DepthConv(256, 128, 3, 1, norm_layer=norm_layer, activation=activation))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                 DepthConv(128, 64, 3, 1, norm_layer=norm_layer, activation=activation))

        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                        Decoder_DepthConv(64, output_nc, kernel_size=7, padding=0, norm_layer=norm_layer, add_last_activation=True))

    def forward(self, input, dlatents):
        x = input

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)
        bot = []
        bot.append(x)
        features = []
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)
            bot.append(x)

        if self.deep:
            y4 = self.up4(x)
            features.append(y4)
            y3 = self.up3(y4)
            features.append(y3)
        else:
            y3 = self.up3(x)
            features.append(y3)
        y2 = self.up2(y3)
        features.append(y2)
        y1 = self.up1(y2)
        features.append(y1)
        y = self.last_layer(y1)
        # y = (y + 1) / 2

        # return y, bot, features, dlatents
        return y

#-----------------------------------------------------------

class Generator_Adain_Upsample_V2(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, n_blocks=3, deep=True,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Generator_Adain_Upsample_V2, self).__init__()

        activation = nn.ReLU(True)

        self.deep = deep

        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                         DepthConv(input_nc, 16, kernel_size=7, padding=0, norm_layer=norm_layer, activation=activation))
        ### downsample
        self.down1 = DepthConv(16, 16, 3, 2, norm_layer=norm_layer, activation=activation)
        self.down2 = DepthConv(16, 32, 3, 2, norm_layer=norm_layer, activation=activation)
        self.down3 = DepthConv(32, 32, 3, 2, norm_layer=norm_layer, activation=activation)
        if self.deep:
            self.down4 = DepthConv(32, 64, 3, 2, norm_layer=norm_layer, activation=activation)

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [ResnetBlock_Adain(64, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

        ### upsample
        if self.deep:
            self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                     DepthConv(64, 32, 3, 1, norm_layer=norm_layer, activation=activation))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                 DepthConv(32, 32, 3, 1, norm_layer=norm_layer, activation=activation))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                 DepthConv(32, 16, 3, 1, norm_layer=norm_layer, activation=activation))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                 DepthConv(16, 16, 3, 1, norm_layer=norm_layer, activation=activation))

        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                        Decoder_DepthConv(16, output_nc, kernel_size=7, padding=0, norm_layer=norm_layer, add_last_activation=True))

    def forward(self, input, dlatents):
        x = input

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)
        bot = []
        bot.append(x)
        features = []
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)
            bot.append(x)

        if self.deep:
            y4 = self.up4(x)
            features.append(y4)
            y3 = self.up3(y4)
            features.append(y3)
        else:
            y3 = self.up3(x)
            features.append(y3)
        y2 = self.up2(y3)
        features.append(y2)
        y1 = self.up1(y2)
        features.append(y1)
        y = self.last_layer(y1)
        # y = (y + 1) / 2

        # return y, bot, features, dlatents
        return y
