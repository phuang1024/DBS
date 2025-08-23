import torch
from torch.nn import init
from scipy.special import gamma as Gamma
from scipy.stats import gennorm
import numpy as np


def gg_init(model, shape=2, xi=2):
    """Generalized Gaussian Initialization for ReLU"""
    # shape for the shape of parameter distribution
    # xi = 1 for Sigmoid or no activation
    # xi = 2 for ReLU
    # xi = 2 / (1 + k^2) for LeakyReLU 
    with torch.no_grad():
        for name, param in model.named_parameters():
            param_device = param.device
            param_dtype = param.dtype
            if len(param.shape) == 2:
                n_dim = param.shape[0]
                alpha = np.sqrt(xi/n_dim*Gamma(1/shape) / Gamma(3/shape))
                gennorm_params = gennorm.rvs(
                    shape, loc=0, scale=alpha, size=param.shape)
                param.data = torch.from_numpy(gennorm_params)
            else:
                if "weight" in name:
                    param.data = torch.ones(param.shape)
                elif "bias" in name:
                    param.data = torch.zeros(param.shape)

            param.data = param.data.to(param_dtype).to(param_device)
