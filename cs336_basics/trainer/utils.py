import torch
from torch import nn
import math


def cross_entropy(logist,target):
    c=logist.max(dim=-1,keepdim=True).values
    shifted=logist-c
    logs=shifted.exp().sum(dim=-1).log()
    front=torch.gather(logist,dim=-1,index=target.unsqueeze(-1)).squeeze(-1)
    loss=c.squeeze(-1)-front+logs
    return loss.mean()


def learning_rate_schedule(t,a_min,a_max,Tw,Tc):
    if t<Tw:
        at=(t/Tw)*a_max
    elif Tw<=t<=Tc:
        theat=((t-Tw)/(Tc-Tw))*math.pi
        at=a_min+0.5*(1+math.cos(theat))*(a_max-a_min)
    else:
        at=a_min
        
    return at
    

def gradient_clipping(params, max_l2):
    eps = 10e-6
    params_with_grad=[p for p in params if p.grad is not None]
    if len(params_with_grad)==0:
        return
    
    norm=torch.sqrt((sum((p.grad ** 2).sum() for p in params_with_grad)))
    if norm > max_l2:
        clip_coef=max_l2/(norm+eps)
        for p in params_with_grad:
            p.grad.mul_(clip_coef)
    

