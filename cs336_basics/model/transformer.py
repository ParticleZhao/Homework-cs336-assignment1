import torch
import torch.nn as nn
from cs336_basics.model.modules import RMSNorm
from cs336_basics.model.modules import Multihead_self_attention
from cs336_basics.model.modules import RotaryPositionEmbedding
from cs336_basics.model.modules import SwiGLU
from cs336_basics.model.modules import Embedding
from cs336_basics.model.modules import Linear
from cs336_basics.model.modules import softmax




class  transformer_block(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theat,max_seq_len,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.norm1=RMSNorm(d_model=self.d_model,device=device,dtype=dtype)
        self.norm2=RMSNorm(d_model=self.d_model,device=device,dtype=dtype)
        self.mha=Multihead_self_attention(d_model=self.d_model,num_heads=self.num_heads,theat=theat,max_seq_len=max_seq_len,
                                          position_embedding=RotaryPositionEmbedding,device=device,dtype=dtype)
        self.ffn=SwiGLU(d_model=self.d_model,d_ff=self.d_ff,device=device,dtype=dtype)
    def forward(self,x,token_position=None):
        if token_position is None:
            token_position=torch.arange(x.size(-2),device=x.device)
        x1=x+self.mha(self.norm1(x),token_position)
        y=x1+self.ffn(self.norm2(x1))
        return y

class Transformer(nn.Module):
    def __init__(self, vocab_size,context_length,num_layers,d_model,d_ff,num_heads,theat,device=None,dtype=None):
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.num_layers=num_layers
        self.d_model=d_model
        self.d_ff=d_ff
        self.embed=Embedding(num_embeddings=self.vocab_size,embedding_dim=self.d_model,device=device,dtype=dtype)
        self.blocks=nn.ModuleList([transformer_block(d_model=self.d_model,num_heads=num_heads,d_ff=self.d_ff,theat=theat,max_seq_len=self.context_length,device=device,dtype=dtype) for _ in range(num_layers)])
        self.norm=RMSNorm(d_model=self.d_model,device=device,dtype=dtype)
        self.out_linear=Linear(self.d_model,self.vocab_size,device=device,dtype=dtype)

    def forward(self,x,token_position=None):
        token=self.embed(x)
        for block in self.blocks:
            token=block(token)
        norm_token=self.norm(token)
        out=self.out_linear(norm_token)
        return out
        
        






