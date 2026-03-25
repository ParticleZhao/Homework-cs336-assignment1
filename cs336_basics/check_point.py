import torch
import torch.nn as nn

def save_checkpoint(model,optimizer,iteration,out):
    params=model.state_dict()
    Adparams=optimizer.state_dict()
    check_point={
        'model':params,
        'optimizer':Adparams,
        'iteration':iteration
    }
    torch.save(check_point,f=out)
    
def load_checkpoint(src,model,optimizer):
    pkt=torch.load(src,map_location='cpu')
    model.load_state_dict(pkt['model'])
    optimizer.load_state_dict(pkt['optimizer'])
    return pkt['iteration']
    

