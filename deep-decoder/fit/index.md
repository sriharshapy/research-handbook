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

**net** : This is the decoder model instance which must be initialized before
hand and pass into the fit function.

**img_noisy_var** :  
