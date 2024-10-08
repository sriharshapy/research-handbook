---
layout: default
title: "Fit"
parent: Deep Decoder
permalink: /deep-decoder/fit/
nav_order: 4
---

# Model
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}


## exp_lr_scheduler

This function adjusts the learning rate of the optimizer. It has a steady decay based on the epoch number.
```python
def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.65**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
```

## helper methods

```python
def np_to_tensor(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)

def np_to_var(img_np, dtype=torch.float32, device='cuda'):
    '''Converts image in numpy.array to torch.Tensor with an additional batch dimension.

    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    # return Variable(np_to_tensor(img_np)[None, :])
    tensor = np_to_tensor(img_np).to(dtype=dtype, device=device)
    return tensor.unsqueeze(0)
```

The original code has the line `return Variable(np_to_tensor(img_np)[None, :])`.
We remove `Variable` because it is depreciate but in the latest torch versions any tensor we create will be by default automatically differentiable. The `[None, :]` is used to create an extra dimension or wrap `[]`
around the tensor data. We will use  `unsqueeze(0)` which has the same functionality.


## fit
Exploring each param of the fit function.

### Denoising

```python
def fit(net,
        img_noisy_var,
        num_channels,
        img_clean_var,
        num_iter=5000,
        LR=0.01,
        OPTIMIZER='adam',
        opt_input=False,
        reg_noise_std=0,
        reg_noise_decayevery=100000,
        mask_var=None,
        apply_f=None,
        lr_decay_epoch=0,
        net_input=None,
        net_input_gen="random",
        find_best=False,
        weight_decay=0,
       ):
```

#### `net` :
This is the decoder model instance which must be initialized before
hand and pass into the fit function.

#### `img_noisy_var` :  

```python
def get_noisy_img(sig=30,noise_same = False):
    sigma = sig/255.
    if noise_same: # add the same noise in each channel
        noise = np.random.normal(scale=sigma, size=img_np.shape[1:])
        noise = np.array( [noise]*img_np.shape[0] )
    else: # add independent noise in each channel
        noise = np.random.normal(scale=sigma, size=img_np.shape)

    img_noisy_np = np.clip( img_np + noise , 0, 1).astype(np.float32)
    img_noisy_var = np_to_var(img_noisy_np).type(dtype)
    return img_noisy_np, img_noisy_var
```

The `img_noisy_var` is a tensor which contains an image which is created by
adding noise to the input image in the case of denoising experiment. It contains the same
data as the `img_noisy_np` which is the other output of `get_noisy_img(args...)`
but with extra added dimension or `[]` wrapped around the data and also `required_grad=True`.

#### `num_channels` :
