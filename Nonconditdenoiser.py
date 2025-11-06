import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_

class Nodenoiser(nn.Module):
    """
    A deep neural network for the reverse process of latent diffusion.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, act_func='tanh', dropout=0.5):
        super(Nodenoiser, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.time_type = time_type
        self.norm = norm
    
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        self.norm1 = nn.LayerNorm(in_dims)
        self.in_layers = nn.Sequential( nn.Linear(in_dims+emb_size, in_dims))
        self.out_layers =   nn.Sequential( nn.Linear(in_dims, in_dims))
        self.dropout = nn.Dropout(dropout)
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
		
        for layer in  self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    

    def forward(self,x,attembed,timesteps,iftraining):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)#
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x, dim=-1)
        x0 = x
        if iftraining:
            x = self.dropout(x)
        h = torch.cat([emb,x], dim=-1)
        h = self.in_layers(h)
        h = self.norm1(h)
        h = torch.tanh(h) 
        h = self.out_layers(h)
        return h
        
       
        
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.#创建正弦时间步嵌入

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

     



