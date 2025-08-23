import torch
from utils.encode.quantizer import LinearQuantizer
import math
from scipy.special import gamma as Gamma
import numpy  as np
import dask.array as da


class DeepShape:
    def __init__(self):
        self.gamma_table = torch.load('utils/gamma_table.pt')
        self.rho_table = torch.load('utils/rho_table.pt')
    
    """estimate GGD parameters"""
    def Calc_GG_params(self, model, adj_minnum = 0):
        #get parameters
        params = []
        for param in model.parameters():
            params.append(param.flatten())
        params = torch.cat(params).detach()
        params_org = params.clone()
        
        # Quantization
        lq = LinearQuantizer(params, 13)
        params = lq.quant(params)   
        
        #sorting
        elements, counts = torch.unique(params, return_counts=True)   
        # dask_params = da.from_array(params.numpy(), chunks=int(1e8))  #if param's size is big
        # elements, counts = da.unique(dask_params, return_counts=True)
        # elements = torch.from_numpy(elements.compute())
        # counts = torch.from_numpy(counts.compute())
        indices = torch.argsort(counts, descending=True)
        elements = elements[indices]
        counts = counts[indices]

        if adj_minnum > 0:
            param_max = torch.min(elements[(counts<=adj_minnum) & (elements>0)]).long()
            # print("param_max", (param_max/(2**13)))
            # print('max_param, num_max_param', (elements[0]/(2**13)), counts[0])
            elements_cut = params_org[torch.abs(params_org)<=(param_max.float()/(2**13))]
        else:
            elements_cut = params_org

        #estimate
        n = len(elements_cut)
        var = torch.sum(torch.pow(elements_cut, 2))
        mean = torch.sum(torch.abs(elements_cut))
        self.gamma_table = self.gamma_table.to(elements_cut.device)
        self.rho_table = self.rho_table.to(elements_cut.device)
        rho = n * var / mean ** 2
        pos = torch.argmin(torch.abs(rho - self.rho_table)).item()
        shape = self.gamma_table[pos].item()
        std = torch.sqrt(var / n)
        beta = math.sqrt(Gamma(1/shape) / Gamma(3/shape))* std
        mu = torch.mean(elements_cut)
        print("mu:", mu)
        print('shape:', shape)
        print('beta',(beta))
        
        return mu, shape, beta
        
    
    """GGD deepshape remap"""
    def GGD_deepshape(self, model, shape_scale=0.8, std_scale=0.6, adj_minnum = 1000): 
        #get parameters
        params = []
        for param in model.parameters():
            params.append(param.flatten())
        params = torch.cat(params).detach()
        params_org = params.clone()
        
        # Quantization
        lq = LinearQuantizer(params, 13)
        params = lq.quant(params)   
        
        #sorting
        elements, counts = torch.unique(params, return_counts=True)
        indices = torch.argsort(counts, descending=True)
        elements = elements[indices]
        counts = counts[indices]
             
        if adj_minnum > 0:
            param_max = torch.min(elements[(counts<=adj_minnum) & (elements>0)]).long()
            elements_cut = params_org[torch.abs(params_org)<=(param_max.float()/(2**13))]
        else:
            elements_cut = params_org
            param_max=0
        
        #estimate org GGD    
        n = len(elements_cut)
        var = torch.sum(torch.pow(elements_cut, 2))
        mean = torch.sum(torch.abs(elements_cut))
        self.gamma_table = self.gamma_table.to(elements_cut.device)
        self.rho_table = self.rho_table.to(elements_cut.device)
        rho = n * var / mean ** 2
        pos = torch.argmin(torch.abs(rho - self.rho_table)).item()
        shape = self.gamma_table[pos].item()
        std = torch.sqrt(var / n)
        beta = math.sqrt(Gamma(1/shape) / Gamma(3/shape))* std
        mu_est = torch.mean(elements_cut)
    
        print("org mu:", mu_est)
        print('org shape:', shape)
        print('org beta',beta)
        
        beta = (beta * (2**13))
        mu_est = int(mu_est*(2**13))
        
        #sorting params in [-param_pax, param_max]  
        if adj_minnum>0:
            adj_indices = torch.nonzero((params>=mu_est-param_max)&(params<=mu_est+param_max), as_tuple=False).squeeze()
            adj_indices = adj_indices[torch.argsort(params[(params>=mu_est-param_max)&(params<=mu_est+param_max)], descending=False)]
            adj_num = len(adj_indices)
        else:
            adj_indices = torch.argsort(params, descending=False)
            adj_num = len(adj_indices)
            
        #remape new GGD
        new_params = params.clone()
        new_shape = shape * shape_scale
        new_beta = beta * std_scale
        if(beta<=0):
            beta=1
        
        x = torch.arange(mu_est-param_max, mu_est+param_max+1, device=params.device)
        new_ratio = -torch.pow(torch.abs(x.float()-mu_est)/new_beta, new_shape)
        new_ratio = torch.exp(new_ratio)
        new_ratio = new_ratio / torch.sum(new_ratio)
        new_num = (adj_num * new_ratio).long()
        num_temp = 0
        for i in range(0, 2*param_max+1):
            new_params[adj_indices[num_temp : num_temp+new_num[i]]]=i+mu_est-param_max
            num_temp += new_num[i]
        new_params=new_params.float()/(2**13)
        
        #modify model parameters
        j=0
        for name, param in model.named_parameters():
            shape=param.data.shape
            param_flatten = torch.flatten(param.data)
            param_flatten = new_params[j: j+len(param_flatten)]
            j+=len(param_flatten)
            param_flatten = param_flatten.reshape(shape)
            param.data= param_flatten
        
        print("new mu:", float(mu_est)/(2**13))
        print('new_shape:', new_shape)
        print('new beta', float(new_beta)/(2**13))
        return float(mu_est)/(2**13), new_shape, float(new_beta)/(2**13)
    
   