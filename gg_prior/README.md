---
title: Gg Prior
emoji: 🧠
colorFrom: yellow
colorTo: indigo
sdk: static
pinned: false
license: cc-by-nc-sa-4.0
short_description: The code of gg prior.
---

# Introduction
- Reference: It Takes a Good Model to Train a Good Model: Generalized Gaussian Priors for Optimized LLMs
- Authors: Jun Wu, Yirong Xiong, Jiangtao Wen, Yuxing Han
- Paper Link: [https://arxiv.org/abs/2506.00486](https://arxiv.org/abs/2506.00486)

This repository provides a complete implementation of the methods described in the corresponding paper. Specifically, we implement the Generalized Gaussian Initialization, DeepShape, and the RF8 floating-point format as proposed in the paper. Furthermore, we adapt and reproduce the BackSlash training algorithm, and incorporate it seamlessly into our framework based on generalized Gaussian priors.


# BackSlash
- Reference: BackSlash: Rate Constrained Optimized Training of Large Language Models
- Authors: Jun Wu, Jiangtao Wen, Yuxing Han
- Paper Link: [https://arxiv.org/abs/2504.16968](https://arxiv.org/abs/2504.16968)

We reproduced the BackSlash training algorithm based on the algorithm diagram provided in the source paper, and assisted us in conducting more in-depth research on generalized Gaussian priors.

In BackSlash, estimating the shape parameters of the model parameter distribution requires looking up the mapping between $\rho(\nu)$ and $\nu$. To achieve this, we precompute the values of $\nu$ and $\rho(\nu)$ over the interval $[0.1,\, 3.0]$ with a step size of $0.01$, and store them in `data/gamma_table.pt` (for $\nu$) and `data/r\_gamma\_table.pt` (for $\rho(\nu)$), respectively.

The code for reproducing the shape parameter estimation and the BackSlash algorithm is stored in the `backslash.py` module. During model training, after each batch iteration, the BackSlash function is invoked to perform rate suppression on the model parameters. After a few epochs of BackSlash training, we further fine-tune the model using several epochs of standard training. This helps the model achieve significantly improved performance while maintaining a low bit rate. The same procedure is applied consistently across all experiments.

# Generalized Gaussian Initialization

The Generalized Gaussian Initialization algorithm is implemented in the file `gg_init.py`. In practice, this initialization function is applied to all linear layers of the model prior to the start of training.

The generalized Gaussian initialization takes two parameters: `shape` and `xi`. 

- `shape` represents the user-specified shape parameter for generalized Gaussian initialization, which affects the parameter distribution during initialization.
- `xi` represents the user-specified activation function coefficient, whose value is determined by the specific type of activation function.

# DeepShape

The implementation of DeepShape is preserved in the file `deepshape.py`. We apply the classical image processing technique "histogram equalization" to adjust the parameter distribution of post-trained models.

While DeepShape effectively compresses parameter bitrate, it inevitably impacts model prediction accuracy. This effect on model performance can be mitigated through a limited number of post-training epochs.

# 8-bit Residual Floating-Point Format

The implementation method of the RF8 floating-point format is documented in the file `rf8.py`. In this paper, we only quantize the model parameters to RF8 format before inference, without considering the training scenario or quantizing the activation values during forward propagation in inference.

In RF8 format, all model parameters will be preserved as only the sum of the first two significant digits in their binary representation. The multiplicative relationship between these two significant digits does not exceed $2^{4}$.


# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
