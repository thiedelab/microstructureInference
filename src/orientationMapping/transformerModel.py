import torch
import torch.nn as nn
from embdding import EmbedLayer
import math
from dataclasses import dataclass

# from transformer_blocks import Encoder
# One can also import Encoder from transformer_blocks.py in my Github repository.
# I copied the code here so that this notebook is self-contained.  

@dataclass
class ModelConfig:
    d_embed: int
    # d_ff is the dimension of the fully-connected  feed-forward layer
    d_ff: int
    # h is the number of attention head
    angle_bin_centers: torch.tensor
    num_bins_braggintensity: int
    num_bins_radialDistance: int
    device: torch.device
    num_feature: int
    h: int
    N_encoder: int
    max_seq_len: int
    dropout: float

def make_model(config):
    model = Transformer(config, config.num_feature).to(config.device)
    # initialize model parameters
    # it seems that this initialization is very important!
    for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_embed, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_embed % h == 0 # check the h number
        self.d_k = d_embed//h
        self.d_embed = d_embed
        self.h = h
        self.WQ = nn.Linear(d_embed, d_embed)
        self.WK = nn.Linear(d_embed, d_embed)
        self.WV = nn.Linear(d_embed, d_embed)
        self.linear = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0) # get batch size
        # 1) Linear projections to get the multi-head query, key and value tensors
        # x_query, x_key, x_value dimension: nbatch * seq_len * d_embed
        # LHS query, key, value dimensions: nbatch * h * seq_len * d_k
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        key   = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        # 2) Attention
        # scores has dimensions: nbatch * h * seq_len * seq_len
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.d_k)
        # 3) Mask out padding tokens and future tokens
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        # p_atten dimensions: nbatch * h * seq_len * seq_len
        p_atten = torch.nn.functional.softmax(scores, dim=-1)
        p_atten = self.dropout(p_atten)
        # x dimensions: nbatch * h * seq_len * d_k
        x = torch.matmul(p_atten, value)
        # x now has dimensions:nbtach * seq_len * d_embed
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.d_embed)
        return self.linear(x) # final linear layer


class ResidualConnection(nn.Module):
  '''residual connection: x + dropout(sublayer(layernorm(x))) '''
  def __init__(self, dim, dropout):
      super().__init__()
      self.drop = nn.Dropout(dropout)
      self.norm = nn.LayerNorm(dim)

  def forward(self, x, sublayer):
      return x + self.drop(sublayer(self.norm(x)))

# I simply let the model learn the positional embeddings in this notebook, since this
# almost produces identital results as using sin/cosin functions embeddings, as claimed
# in the original transformer paper. Note also that in the original paper, they multiplied
# the token embeddings by a factor of sqrt(d_embed), which I do not do here.

class Encoder(nn.Module):
    '''Encoder = token embedding + positional embedding -> a stack of N EncoderBlock -> layer norm'''
    def __init__(self, config):
        super().__init__()
        self.d_embed = config.d_embed
        # self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)

        self.embed = EmbedLayer(
                                self.d_embed,
                                config.angle_bin_centers,
                                config.num_bins_braggintensity,
                                config.num_bins_radialDistance,
                                config.device
                                )
        # self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.N_encoder)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, input, mask=None):
        x = self.embed(input)
        x = self.dropout(x)
        # x = self.tok_embed(input)
        # x_pos = self.pos_embed[:, :x.size(1), :]
        # x = self.dropout(x + x_pos)
        for layer in self.encoder_blocks:
            x = layer(x, mask)
        return self.norm(x)


class EncoderBlock(nn.Module):
    '''EncoderBlock: self-attention -> position-wise fully connected feed-forward layer'''
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.atten = MultiHeadedAttention(config.h, config.d_embed, config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual1 = ResidualConnection(config.d_embed, config.dropout)
        self.residual2 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, x, mask=None):
        # self-attention
        x = self.residual1(x, lambda x: self.atten(x, x, x, mask=mask))
        # position-wise fully connected feed-forward layer
        return self.residual2(x, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, config, num_feature):
        super().__init__()
        self.encoder = Encoder(config)
        self.MLP_head_0 = nn.Sequential(
            nn.Linear(config.d_embed, int(config.d_embed / 2)),
            # nn.LayerNorm(int(config.d_embed / 2)),
            nn.GELU(),
            nn.Linear(int(config.d_embed / 2), int(config.d_embed / 4)),
            # nn.LayerNorm(int(config.d_embed / 4)),
            nn.GELU(),
            nn.Linear(int(config.d_embed / 4), int(config.d_embed / 8)),
            # nn.LayerNorm(int(config.d_embed / 8)),
            nn.GELU(),
            nn.Linear(int(config.d_embed / 8), num_feature)
        )
        
        self.MLP_head_1 = nn.Sequential(
            nn.Linear(config.d_embed, int(config.d_embed / 2)),
            # nn.LayerNorm(int(config.d_embed / 2)),
            nn.GELU(),
            nn.Linear(int(config.d_embed / 2), int(config.d_embed / 4)),
            # nn.LayerNorm(int(config.d_embed / 4)),
            nn.GELU(),
            nn.Linear(int(config.d_embed / 4), int(config.d_embed / 8)),
            # nn.LayerNorm(int(config.d_embed / 8)),
            nn.GELU(),
            nn.Linear(int(config.d_embed / 8), 1)
        )


    def forward(self, x, pad_mask=None):
        x = self.encoder(x, pad_mask)
        if pad_mask is not None:
            reshaped_mask = pad_mask.reshape(pad_mask.shape[0], pad_mask.shape[-1], 1)
            reshaped_mask = torch.logical_not(reshaped_mask).float()
            x = x * reshaped_mask
            x = torch.sum(x, dim = 1)
            reshaped_mask_sum = torch.sum(reshaped_mask, dim = 1)
            x_final = x / reshaped_mask_sum
        else:
            x_final = torch.mean(x,-2)


        return  self.MLP_head_0(x_final), self.MLP_head_1(x_final)
