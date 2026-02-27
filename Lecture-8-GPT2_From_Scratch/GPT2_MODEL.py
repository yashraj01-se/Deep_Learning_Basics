import math # For Scaling Attention Mechanism 
import torch # Pytorch
import torch.nn as nn # Module for linear,layernorm,etc
from dataclasses import dataclass # Used to define a lightweight config container.
from torch.nn import functional as F # Stateless Function like softmax,etc
from transformers import GPT2LMHeadModel # GPT2 from huggingFace for pretrained weights

#######################################################################################################################

@dataclass # @dataclass auto-generates:__init__,__repr__
class GPTconfig:
    block_size:int=1024 #Max Sequence length
    vocab_size:int=50257 #50,000 merges+256characters+1 special token <|endoftoken|>
    n_layer:int=12 #number of layers
    n_head:int=12 #number of heads
    n_embed:int=768 #emdding dimension

class CausalSelfAttention(nn.Module): # Implementing masked multi-head self-attention.
    def __init__(self,config): # Defining function 
        super().__init__() # Constructor
        assert config.n_embed%config.n_head==0 # So we can evenly distribute the embeddings across heads

        # Instead of Computing K,Q,V Projections seperately, we Compute one Matrix [Q|K|V]:
        self.c_attn=nn.Linear(config.n_embed,3*config.n_embed)

        # Final Output Projection after concatenating heads:
        self.c_proj=nn.Linear(config.n_embed,config.n_embed)

        # Regularization
        self.n_head=config.n_head
        self.n_embed=config.n_embed

        #According to the GPT2 (Register tensors as the part of model)
        #(torch.tril.....(For, Applying Actual masking such that (token i​ can only attend to j≤i))):
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                            .view(1,1,config.block_size,config.block_size)) # To broadcast

    def forward(self,x):
        B,T,C=x.size() #Input shape is (B,T,768)
        qkv=self.c_attn(x) # Shape is (B,T,2304) #Single QKV matrix for better computation
        q,k,v=qkv.split(self.n_embed,dim=2) # Each will become (B,T,768) 
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #Breaks 768 into 12 heads each of 64 dim so shape (B,T,12,64) after transpose (B,12,T,64)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #Breaks 768 into 12 heads each of 64 dim so shape (B,T,12,64) after transpose (B,12,T,64)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #Breaks 768 into 12 heads each of 64 dim so shape (B,T,12,64) after transpose (B,12,T,64)
        att=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1))) #Computes QKT/root(dk) Shape(B,12,T,T) so that each token attends to every token ​
        att=att.masked_fill(self.bias[:,:,:T,:T]==0,float("-inf")) # Replaces future Tokens with (-inf) 
        att=F.softmax(att,dim=-1) # For Probability distribution (weights sum equal to 1)
        y=att@v  # Chages are made in V matix , shape (B,12,T,64)
        y=y.transpose(1,2).contiguous().view(B,T,C) # Back to shape (B,T,768)
        y=self.c_proj(y) # Linear transformation across embedding dimension.
        return y # Returning statement

class MLP(nn.Module): # Multi Layer perceptron Definition (Feed Forward Network of Transformers)
    def __init__(self,config):
        super().__init__() 
        self.c_fc=nn.Linear(config.n_embed,4*config.n_embed) # Expands dimension:768→3072
        self.gelu=nn.GELU(approximate='tanh') # Activation function 
        self.c_proj=nn.Linear(4*config.n_embed,config.n_embed) # Back to original shape (3072→768)

    def forward(self,x): # Forward function of FFN layer
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class Block(nn.Module): # Generalize Transformer Block
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embed) # Normalization before Self-Attention layer as per GPT-2 paper 
        self.attn=CausalSelfAttention(config) # Attention here is like a reduce function
        self.ln_2=nn.LayerNorm(config.n_embed) # Second Normalization layer
        self.mlp=MLP(config) # MLP here is like a mapping Function

    def forward(self,x): 
        x=x+self.attn(self.ln_1(x)) # First Residual Branch
        x=x+self.mlp(self.ln_2(x)) # Second Residual Branch
        return x
        # Together they Build a MapReduce Function

class GPT2(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict( # Whole Transformer Decoder Block (output embeddings->positional embeddings->transformer blocks->layernorm->linear layer->)
            wte=nn.Embedding(config.vocab_size,config.n_embed), # Token embeddings (output embeddings) Maps token id → 768 vector.
            wpe=nn.Embedding(config.block_size,config.n_embed), # Positional embeddings
            h=nn.ModuleList([Block(config)for _ in range(config.n_layer)]), # Stack 12 identical Transformer block blocks.
            ln_f=nn.LayerNorm(config.n_embed) # Added layernorm layer according to GPT2
        ))
        self.lm_head=nn.Linear(config.n_embed,config.vocab_size,bias=False) # Maps embedding → logits over vocabulary.
        self.lm_head.weight = self.transformer.wte.weight # Shares input and output embedding matrix.

    def forward(self, idx):
        B, T = idx.size() # Each element is a token ID (integer in [0, vocab_size]) We separate batch size and time dimension.
        assert T <= self.config.block_size # Prevents overflow of:Positional embeddings,Causal mask
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0) # Create a list of [0,1,2,...T-1] 
        tok_emb = self.transformer.wte(idx)  # Input-> (B,T) output-> (B,T,768) #Each token ID becomes a learned vector.
        pos_emb = self.transformer.wpe(pos) # Input-> (1,T) output-> (1,T,768) #learned positional encoding.
        x = tok_emb + pos_emb # Broadcasting (B,T,768)+(1,T,768)-> 
        for block in self.transformer.h: # Each transformer block: x=x+Attention(LN(x)) 𝑥=𝑥+𝑀𝐿𝑃(𝐿𝑁(𝑥))
            x = block(x)

        x = self.transformer.ln_f(x) # GPT-2 applies one final normalization before projection.
        logits = self.lm_head(x) # Input-> (B,T,768) -> (B,T,50257) No softmax here because the logits are expected to be raw.

        return logits

    @classmethod # method belongs to class, not instance.
    def from_pretrained(cls,model_type): #cls → refers to GPT2 class itself. Allows:model = GPT2.from_pretrained("gpt2"),Instead of needing an existing object.
        """Loading Pretrained Weights from Hugging Face"""
        assert model_type in {"gpt2","gpt2-medium","gpt2-large","gpt2-xl"} #Prevents mismatched configs.
        print(f"Loading Pretrained weigts of model:{model_type}")
        
        config_args={ #Each GPT-2 variant differs only in:number of layers,number of heads,embedding dimension->Vocab size and block size remain constant.
            "gpt2":dict(n_layer=12,n_head=12,n_embed=768), #124M parameters
            "gpt2-medium":dict(n_layer=24,n_head=16,n_embed=1024), #350 paramerters
            "gpt2-large":dict(n_layer=36,n_head=20,n_embed=1280), # 774M parameters 
            "gpt2-xl":dict(n_layer=48,n_head=25,n_embed=1600) #1550M parameters
        }[model_type]
        config_args['vocab_size']=50257 #Explicitly sets shared properties.
        config_args['block_size']=1024 #Explicitly sets shared properties.
        config=GPTconfig(**config_args) #Creates your custom GPT2 instance.
        model=GPT2(config=config) #At this moment:Weights are randomly initialized.Structure matches GPT-2,But not pretrained yet.
        sd=model.state_dict() #sd = OrderedDict of:{"transformer.wte.weight": tensor,"transformer.wpe.weight": tensor,...} thse are the our model parameters.
        sd_keys=sd.keys()

        #Loading pretrained weights:
        model_hf=GPT2LMHeadModel.from_pretrained(model_type) #Now we have HuggingFace’s parameter dictionary.
        sd_hf=model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys=[k for k in sd_keys if not k.endswith(".attn.bias")] #ingoring attn.bias
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")] #ignoring attn.masked.bias
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith(".attn.bias")] #ignoring attn.bias
        #because They are buffers,Not learned parameters->Not needed for copying

        transposed = ["attn.c_attn.weight","attn.c_proj.weight","mlp.c_fc.weight","mlp.c_proj.weight"]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to
        # this means that we have to transpose these weights when we import them
        # Notice weight Shapes->(in_features,out_features) but pytorch nn.Linear Layer expect(out_features,in_features that's why TRANSPOSED
        assert set(sd_keys_hf) == set(sd_keys), f"mismatched keys: {len(sd_keys_hf)}" #Ensures same number of parameters.
        for k in sd_keys_hf: #COPYING PARAMETERS
            if any(k.endswith(w) for w in transposed):
         # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
        # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                         sd[k].copy_(sd_hf[k])
        return model
    
######################################################################################################################

# CHECKER:

hf = GPT2LMHeadModel.from_pretrained("gpt2")
custom = GPT2.from_pretrained("gpt2")

idx = torch.randint(0,50257,(1,10))

with torch.no_grad():
    out1 = hf(idx).logits
    out2 = custom(idx)

print(torch.allclose(out1, out2, atol=1e-3))