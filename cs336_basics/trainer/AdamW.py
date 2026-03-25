import torch
import math





class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                g = p.grad.data
                if len(state) == 0:
                    state['m'] = torch.zeros_like(g)
                    state['v'] = torch.zeros_like(g)
                    state['t'] = 0
                
                state['t'] += 1
                t = state['t']
                
                state['m'].mul_(b1).add_(g, alpha=1 - b1)        
                state['v'].mul_(b2).addcmul_(g, g, value=1 - b2) 
                
                a = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                
                p.data.addcdiv_(state['m'], state['v'].sqrt().add_(eps), value=-a)  
                p.data.mul_(1 - lr * weight_decay)                                  
        return loss