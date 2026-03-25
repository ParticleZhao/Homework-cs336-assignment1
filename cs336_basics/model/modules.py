import torch.nn as nn # nn.Parameter, nn.Module, nn,ModuleList,nn.Squential
import torch
from einops import rearrange,einsum

class Linear(nn.Module):
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.device=device
        self.dtype=dtype
        self.weight=nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))
        self._init_weights()

    def forward(self,x):
        return einsum(x,self.weight,"... d_in, d_out d_in -> ... d_out")

    def _init_weights(self):
        std=(2/(self.in_features+self.out_features))**0.5
        torch.nn.init.trunc_normal_(self.weight,mean=0,std=std,a=-3*std,b=3*std)    
        

class Embedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.weight=nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        self._init_embedding()
    def forward(self,token_ids):
        return self.weight[token_ids.long()]
    
    def _init_embedding(self):
        torch.nn.init.trunc_normal_(self.weight,mean=0,std=1,a=-3,b=3)


class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.device=device
        self.dtype=dtype
        self.weight=nn.Parameter(torch.ones(d_model,device=device,dtype=torch.float32))
    
    def forward(self,x:torch.Tensor):
        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms=x.pow(2).mean(-1,keepdim=True)
        rms1=torch.sqrt(rms+self.eps)
        result=(x/rms1)*self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model,d_ff,device=None,dtype=None):
        super().__init__()
        self.d_model=int(d_model)
        self.device=device
        self.dtype=dtype
        self.d_ff=int(d_ff)
        self.W1=nn.Parameter(torch.empty(d_ff,d_model,device=device,dtype=dtype))
        self.W3=nn.Parameter(torch.empty(d_ff,d_model,device=device,dtype=dtype))
        self.W2=nn.Parameter(torch.empty(d_model,d_ff,device=device,dtype=dtype))
        self._init_W()


    def forward(self,x):
        f1=einsum(x,self.W1,' ... d, d_ff d -> ... d_ff')
        f2=einsum(x,self.W3,' ... d, d_ff d -> ... d_ff')
        gate=f1*(1/(1+torch.exp(-f1)))
        f3=f2*gate
        out=einsum(f3,self.W2,'... d_ff, d d_ff -> ... d')
        return out

    def _init_W(self):
        std=(2/(self.d_ff+self.d_model))**0.5
        torch.nn.init.trunc_normal_(self.W1,mean=0,std=std,a=-3*std,b=3*std)
        torch.nn.init.trunc_normal_(self.W2,mean=0,std=std,a=-3*std,b=3*std)
        torch.nn.init.trunc_normal_(self.W3,mean=0,std=std,a=-3*std,b=3*std)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theat:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        freqs=1/(theat**((torch.arange(0,d_k,2))/d_k))
        postions=torch.arange(max_seq_len)
        angel=torch.outer(postions.float(),freqs)
        self.register_buffer('cos_cached',angel.cos(),persistent=False)
        self.register_buffer('sin_cached',angel.sin(),persistent=False)


    def forward(self,x,token_positions):
        cos=self.cos_cached[token_positions]
        sin=self.sin_cached[token_positions]
        cos2=torch.repeat_interleave(cos,2,dim=-1)
        sin1=torch.repeat_interleave(sin,2,dim=-1)
        x_r=x*cos2+self.rotate_half(x)*sin1
        return x_r

    def rotate_half(self,x):
        x_odd=x[...,0::2]
        x_even=-x[...,1::2]
        x_all=torch.stack([x_even,x_odd],dim=-1)
        y=torch.flatten(x_all,start_dim=-2)
        return y

def softmax(x,dim:int):
    x=x-torch.max(x,dim=dim,keepdim=True).values
    exp_X=torch.exp(x)
    sum_x=torch.sum(exp_X,dim=dim,keepdim=True)
    return  exp_X/sum_x


def Scaled_Dot_Product_Attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    qk=einsum(q,k,'... seq_n d, ... seq_m d -> ... seq_n seq_m')/(q.size(-1)**0.5)
    if mask is not None:
        qk=qk.masked_fill(~mask,float('-inf'))
    m_qk=softmax(qk,dim=-1)
    qkv=einsum(m_qk,v,'... seq_n seq_m, ... seq_m d_v -> ... seq_n d_v')
    return qkv

class Multihead_self_attention(nn.Module):
    def __init__(self, d_model,num_heads,theat:float,max_seq_len:int,position_embedding:nn.Module=None,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.head=d_model//num_heads
        self.q=Linear(self.d_model,self.d_model,device=device,dtype=dtype)
        self.k=Linear(self.d_model,self.d_model,device=device,dtype=dtype)
        self.v=Linear(self.d_model,self.d_model,device=device,dtype=dtype)
        self.o=Linear(self.d_model,self.d_model,device=device,dtype=dtype)
        self.pe=None
        if theat is not None and max_seq_len is not None and position_embedding is not None:
            self.pe=position_embedding(theat=theat,d_k=self.head,max_seq_len=max_seq_len,device=device)
        
    
    def causalmask(self,seq_len):
        mask=torch.tril(torch.ones([seq_len,seq_len])).bool()
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self,x,token_position=None):
        wq=self.q(x)
        wk=self.k(x)
        wv=self.v(x)
        wq_n=rearrange(wq,'b s (h d)-> b h s d',h=self.num_heads)
        wk_n=rearrange(wk,'b s (h d)-> b h s d',h=self.num_heads)
        wv_n=rearrange(wv,'b s (h d)-> b h s d',h=self.num_heads)
        if self.pe is not None:
            if token_position is None:
                token_position=torch.arange(x.size(-2),device=x.device)
            wq_n=self.pe(wq_n,token_position)
            wk_n=self.pe(wk_n,token_position)
        mask=self.causalmask(wq_n.size(-2)).to(wq_n.device)
        att=Scaled_Dot_Product_Attention(wq_n,wk_n,wv_n,mask)
        y=rearrange(att,'b h s d -> b s (h d)')
        out=self.o(y)
        return out        
        

        