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

`net` :
This is the decoder model instance which must be initialized before
hand and pass into the fit function.

`img_noisy_var` :  

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

`num_channels` : This is the same param we pass into the decoder model `num_channels_up`.
In fit method it is used to compute the shape of `net_input` which is initialized with zeros
having shape of `[1,num_channels[0], width, height]`. `net_input` is computed only
in the scenarios where it is not explicitly pass as an arg. In case of Denoising
experiment is will not be passed.

`img_clean_var` : This contains the data of the original image without any noise.
The data is wrapped in a tensor with `required_grad=True` and has an extra
dimension because it is wrapped with `[]`.

`num_iter` : This hold the number of iterations the forward and backward passes
occur during the execution of the fit model.

`LR`: This the learning rate that is fed into the optimizer.

`OPTIMIZER` : This will set the kind of optimizer that the fit method will
employ during the course of the iterations defined by `num_iter`. The fit method
will accept only these three types of optimizers `SGD`, `adam`, `LBFGS`.

`opt_input` : This is a flag, when set to true will add the tensor `net_input`
into the list of params along side of decoder params that are set to optimizer
over the several iterations defined by `num_iter`.

`reg_noise_std` : This hold the standard deviation which is multiplied to a noise
generated using a normal distribution with `mean=0` and `std=1`. This noise Tensor
has the same shape as the `net_input`. In every iteration this noise multiplied with
`reg_noise_std` is added to the `net_input`.

`reg_noise_decayevery` : This is the frequency at which the `reg_noise_std` is
decayed.

`mask_var` : A mask that will be applied to output and the target data to mask
out parts of the image or data. This is optional.

`apply_f` : This holds a function that will be applied to the output right before
computing the loss.

`lr_decay_epoch` : This contributes to the decay of learning rate. The decay of
learning rate is exponential in nature. The rate at which the decay must happen can
be tuned using this param. The param is passed into method `exp_lr_scheduler(args..)`
from fit.

`net_input` : This holds the input to the decoder model. This tensor can be
passed as an argument into the fit method. In case its not passed, the method
creates one with same shape as the expected input to the decoder model. The data
will be sampled from a uniform distribution between 0 to 1.

`net_input_gen` : This param is not used anywhere in the fit method. Ignore it.

`find_best` : This is a flag, when set to true will look out for the best model
during the iteration. When the loss is decreased by 0.5 % it is considered better
than its predecessor.

`weight_decay` : This is the weight decay param that is passed into the optimizer.
It will be passed into only `adam` and `SDG` optimizers. In other cases it will
be ignored.
