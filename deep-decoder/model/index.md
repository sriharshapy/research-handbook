---
layout: default
title: "Model"
parent: Deep Decoder
permalink: /deep-decoder/model/
nav_order: 3
---

# Model
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}


## Model input and output shapes

This configuration generated using [nnsight](https://nnsight.net/) library.


```
layer :  Sequential(
  (0): ReflectionPad2d((0, 0, 0, 0))
  (1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
)
input shape : torch.Size([1, 128, 128, 128])
output shape : torch.Size([1, 128, 128, 128])


layer :  Upsample(scale_factor=2.0, mode='bilinear')
input shape : torch.Size([1, 128, 128, 128])
output shape : torch.Size([1, 128, 256, 256])


layer :  ReLU()
input shape : torch.Size([1, 128, 256, 256])
output shape : torch.Size([1, 128, 256, 256])


layer :  BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
input shape : torch.Size([1, 128, 256, 256])
output shape : torch.Size([1, 128, 256, 256])


layer :  Sequential(
  (0): ReflectionPad2d((0, 0, 0, 0))
  (1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
)
input shape : torch.Size([1, 128, 256, 256])
output shape : torch.Size([1, 128, 256, 256])


layer :  Upsample(scale_factor=2.0, mode='bilinear')
input shape : torch.Size([1, 128, 256, 256])
output shape : torch.Size([1, 128, 512, 512])


layer :  BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
input shape : torch.Size([1, 128, 512, 512])
output shape : torch.Size([1, 128, 512, 512])


layer :  Sequential(
  (0): ReflectionPad2d((0, 0, 0, 0))
  (1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
)
input shape : torch.Size([1, 128, 512, 512])
output shape : torch.Size([1, 128, 512, 512])


layer :  Upsample(scale_factor=2.0, mode='bilinear')
input shape : torch.Size([1, 128, 512, 512])
output shape : torch.Size([1, 128, 1024, 1024])


layer :  BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
input shape : torch.Size([1, 128, 1024, 1024])
output shape : torch.Size([1, 128, 1024, 1024])


layer :  Sequential(
  (0): ReflectionPad2d((0, 0, 0, 0))
  (1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
)
input shape : torch.Size([1, 128, 1024, 1024])
output shape : torch.Size([1, 128, 1024, 1024])


layer :  BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
input shape : torch.Size([1, 128, 1024, 1024])
output shape : torch.Size([1, 128, 1024, 1024])


layer :  Sequential(
  (0): ReflectionPad2d((0, 0, 0, 0))
  (1): Conv2d(128, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
)
input shape : torch.Size([1, 128, 1024, 1024])
output shape : torch.Size([1, 3, 1024, 1024])


layer :  Sigmoid()
input shape : torch.Size([1, 3, 1024, 1024])
output shape : torch.Size([1, 3, 1024, 1024])
```
### Code for the above

Assuming that you are executing in google colab
```
!pip install nnsight -q
```

If you are not using colab install the necessary dependencies mentioned below.

```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from nnsight import NNsight
import nnsight
```

Revamped code of deep decoder to build the model for interpretability and studying the code.

```python
class DeepDecoder(nn.Module):
    def __init__(
        self,
        num_output_channels = 3,
        num_channels_up = [128]*5,
        filter_size_up = 1,
        activation_function = nn.ReLU(),
        need_sigmoid = True,
        pad = 'reflection',
        upsample_first = True,
        upsample_mode = 'bilinear',
        bn_before_act = False,
        bn_affine = True):

        super(DeepDecoder, self).__init__()
        '''
        Adding last two layers to the decoder with same last layer size given as input
        Example : if input is [128,128] it is transformed to [128,128,128,128]
        '''
        self.num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
        # layers
        self.n_scales = len(num_channels_up)

        '''
        Need a value for each layer which must have length of .
        Example: [3,3,3]
        '''
        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
          filter_size_up = [filter_size_up] * self.n_scales

        self.layers = nn.Sequential()

        for i in range(len(num_channels_up)-1):

            if upsample_first:
                self.layers.append(self.conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad))
                if upsample_mode!='none' and i != len(num_channels_up)-2:
                    self.layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))
            else:
                if upsample_mode!='none' and i!=0:
                    self.layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))
                self.layers.append(self.conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad))

            if i != len(num_channels_up)-1:
                if(bn_before_act):
                    self.layers.append(nn.BatchNorm2d( num_channels_up[i+1] ,affine=bn_affine))
                self.layers.append(activation_function)
                if(not bn_before_act):
                    self.layers.append(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))

        self.layers.append(self.conv( num_channels_up[-1], num_output_channels, 1, pad=pad))
        if need_sigmoid:
            self.layers.append(nn.Sigmoid())

    def conv(self, in_f, out_f, kernel_size, stride=1, pad='zero'):
        padder = None
        to_pad = int((kernel_size - 1) / 2)
        if pad == 'reflection':
            padder = nn.ReflectionPad2d(to_pad)
            to_pad = 0

        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

        layers = filter(lambda x: x is not None, [padder, convolver])
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

```

Using GPU and a sample initialization

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DeepDecoder(
    num_output_channels = 3,
    num_channels_up = [128]*5,
    upsample_first = True
).to(device)
```

Wrapping Pytorch model with nnsight and using its trace functionality.
Do read about [python contexts](https://book.pythontips.com/en/latest/context_managers.html)
and nnsight doc to understand the interventions in detail.

```python
m = NNsight(net)
input_tensor = torch.randn(1, 128, 128, 128).to(device)

with m.trace(input_tensor) as tracer:
  OUT = []
  for x in m.layers:
    input_dim = x.input.shape.save()
    output_dim = x.output.shape.save()
    OUT.append((x,input_dim,output_dim))
```

Print it
```python
for x in OUT:
  print("layer : ",x[0])
  print("input shape :",x[1])
  print("output shape :",x[2])
  print("\n")
```
