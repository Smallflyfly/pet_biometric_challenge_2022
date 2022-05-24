#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/5/17 18:20 
"""
from typing import Type, Any, Callable, Union, List, Optional

import torch
from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls


class MyResNet(ResNet):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, training=True):
        super(MyResNet, self).__init__()
        self.training = training

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # if not self.training:
        #     return x
        x = self.fc(x)

        return x


def _my_resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = MyResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def my_resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _my_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                      **kwargs)
