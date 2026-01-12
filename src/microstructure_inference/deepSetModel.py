import torch
import torch.nn as nn
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
    intensity_bin_centers: torch.tensor
    num_bins_radialDistance: int
    device: torch.device
    num_feature: int
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

class absolutePositionEmbedding(nn.Module):
    def __init__(self, num_bins, embed_dim):
        super().__init__()
        self.embed_dim  = embed_dim

        positions = torch.arange(num_bins).reshape(-1, 1)                       # N  ->  N, 1
        pos_embedding = self.generate_sinusoidal1D(positions)                   # 1, N, E//2

        self.register_buffer("pos_embedding", pos_embedding)                        # Register_buffer for easy switching of device

    def generate_sinusoidal1D(self, sequence):
        # Denominator
        denominator = torch.pow(10000, torch.arange(0, self.embed_dim, 2) / self.embed_dim)   # E//4                     Denominator used to produce sinusoidal equation
        # print("denominator", denominator)

        # Create an empty tensor and fill with sin and cos values as per sinusoidal embedding equation
        pos_embedding = torch.zeros(1, sequence.shape[0], self.embed_dim)                     # 1, N, E               Used to store positional embedding for x-axis variations
        denominator = sequence / denominator                                                  # N, 1 / (E//2)  ->  N, E//2
        pos_embedding[:, :, ::2]  = torch.sin(denominator)                                    # Fill positional embedding's even dimensions with sin values
        pos_embedding[:, :, 1::2] = torch.cos(denominator)                                    # Fill positional embedding's odd dimensions with cos values
        return pos_embedding.reshape(pos_embedding.shape[1], pos_embedding.shape[2])          # return shape N, E

class directioPositionEmbedding_A(nn.Module):
    def __init__(self, angle_bin_centers, embed_dim, device, num_trainableVec = 11):
        super().__init__()
        # k = int( (num_trainableVec - 1) / 2). If num_trainableVec == 9, k = ((9 - 1) / 2) = 4
        self.embed_dim  = embed_dim
        self.num_trainableVec = num_trainableVec
        self.torch_cosine_angle_library, self.torch_sine_angle_library = self.generate_directional_library(angle_bin_centers)
        self.pos_embedding_learnable_bias = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=True)                                    # 1, E Learnable Positional Embedding
        self.pos_embedding_learnable_cosi = nn.Parameter(torch.zeros(int((num_trainableVec - 1) / 2), embed_dim), requires_grad=True)      # k, E Learnable Positional Embedding
        self.pos_embedding_learnable_sine = nn.Parameter(torch.zeros(int((num_trainableVec - 1) / 2), embed_dim), requires_grad=True)      # k, E Learnable Positional Embedding
        self.torch_cosine_angle_library = self.torch_cosine_angle_library.to(device = device)
        self.torch_sine_angle_library = self.torch_sine_angle_library.to(device = device)

    def generate_directional_library(self, angle_bin_centers):
        torch_cosine_angle_library = []
        torch_sine_angle_library = []
        for k in range(1, int((self.num_trainableVec - 1)/2) + 1, 1):
            torch_cosine_angle_library.append(torch.cos(k * angle_bin_centers))
            torch_sine_angle_library.append(torch.sin(k * angle_bin_centers))
        torch_cosine_angle_library = torch.vstack(torch_cosine_angle_library)                      ## k, 360
        torch_sine_angle_library = torch.vstack(torch_sine_angle_library)                          ## k, 360 
        return torch.permute(torch_cosine_angle_library, (1,0)), torch.permute(torch_sine_angle_library, (1,0)) ## 360, k 

    def forward(self, x):
        ### x: B * S where B is number of batch S is number of sequence
        cosSliced = self.torch_cosine_angle_library[x]                                      ## B * S, k 
        sinSliced = self.torch_sine_angle_library[x]                                        ## B * S, k 
        return torch.matmul(cosSliced, self.pos_embedding_learnable_cosi) + torch.matmul(sinSliced, self.pos_embedding_learnable_sine) + self.pos_embedding_learnable_bias # B * S, E

class directioPositionEmbedding_I(nn.Module):
    def __init__(self, intensity_bin_centers, embed_dim, device, num_trainableVec = 9):
        super().__init__()
        # k = int( (num_trainableVec - 1) / 2). If num_trainableVec == 9, k = ((9 - 1) / 2) = 4
        self.embed_dim  = embed_dim
        self.num_trainableVec = num_trainableVec
        self.torch_cosine_angle_library, self.torch_sine_angle_library = self.generate_directional_library(intensity_bin_centers)
        self.pos_embedding_learnable_bias = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=True)                                    # 1, E Learnable Positional Embedding
        self.pos_embedding_learnable_cosi = nn.Parameter(torch.zeros(int((num_trainableVec - 1) / 2), embed_dim), requires_grad=True)      # k, E Learnable Positional Embedding
        self.pos_embedding_learnable_sine = nn.Parameter(torch.zeros(int((num_trainableVec - 1) / 2), embed_dim), requires_grad=True)      # k, E Learnable Positional Embedding
        self.torch_cosine_angle_library = self.torch_cosine_angle_library.to(device = device)
        self.torch_sine_angle_library = self.torch_sine_angle_library.to(device = device)

    def generate_directional_library(self, intensity_bin_centers):
        torch_cosine_angle_library = []
        torch_sine_angle_library = []
        for k in range(1, int((self.num_trainableVec - 1)/2) + 1, 1):
            torch_cosine_angle_library.append(torch.cos(torch.pi * k * intensity_bin_centers))
            torch_sine_angle_library.append(torch.sin(torch.pi * k * intensity_bin_centers))
        torch_cosine_angle_library = torch.vstack(torch_cosine_angle_library)                      ## k, 360
        torch_sine_angle_library = torch.vstack(torch_sine_angle_library)                          ## k, 360 
        return torch.permute(torch_cosine_angle_library, (1,0)), torch.permute(torch_sine_angle_library, (1,0)) ## 360, k 

    def forward(self, x):
        ### x: B * S where B is number of batch S is number of sequence

        cosSliced = self.torch_cosine_angle_library[x]                                      ## B * S, k 
        sinSliced = self.torch_sine_angle_library[x]                                        ## B * S, k 
        return torch.matmul(cosSliced, self.pos_embedding_learnable_cosi) + torch.matmul(sinSliced, self.pos_embedding_learnable_sine) + self.pos_embedding_learnable_bias # B * S, E


        # self.register_buffer("pos_embedding", pos_embedding)                        # Register_buffer for easy switching of device

class EmbedLayer(nn.Module):
    def __init__(self, embed_dim, angle_bin_centers, intensity_bin_centers, num_bins_radialDistance, device, dropout=0.0):
        super().__init__()
        self.R_embedding = absolutePositionEmbedding(num_bins_radialDistance, embed_dim)                      # 1D Sinusoidal Positional Embedding
        self.A_embedding = directioPositionEmbedding_A(angle_bin_centers, embed_dim, device)                            # directional positional embedding
        self.I_embedding = directioPositionEmbedding_I(intensity_bin_centers, embed_dim, device)                            # directional positional embedding
        # self.cls_token     = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)                     # Classification Token
        self.dropout       = nn.Dropout(dropout)

        # nn.init.trunc_normal_(self.conv1.weight, mean=0.0, std=0.02)
        # nn.init.constant_(self.conv1.bias, 0)
        # nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02)


    def forward(self, x):
        ## x.shape: B, S, 3. Here, 3 corresponds to r - theta - I
        B = x.shape[0]
        S = x.shape[1]
                

        x_r = x.reshape(B * S, x.shape[2])                                         # x : B * S, 3
        
        R_embedding = self.R_embedding.pos_embedding[x_r[:, 0]]                    # R_embdding: B * S, E
        A_embedding = self.A_embedding(x_r[:, 1])                                # A_embdding: B * S, E
        I_embedding = self.I_embedding(x_r[:, 2])                                # I_embdding: B * S, E

        x_r = R_embedding + A_embedding + I_embedding                              # B * S, E
        x_r = x_r.reshape(B, S, R_embedding.shape[1])                                # x : B, S, E


        x_r = self.dropout(x_r)
        return x_r



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
                                config.intensity_bin_centers,
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
        for layer in self.encoder_blocks:
            x = layer(x, mask)
        return self.norm(x)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual1 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, x, mask=None):
        return self.residual1(x, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, config, num_feature):
        super().__init__()
        self.encoder = Encoder(config)
        
        self.feed_forward1 = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual1 = ResidualConnection(config.d_embed, config.dropout)
        
        self.feed_forward2 = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual2 = ResidualConnection(config.d_embed, config.dropout)
        
        self.feed_forward3 = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual3 = ResidualConnection(config.d_embed, config.dropout)
        
        self.feed_forward4 = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual4 = ResidualConnection(config.d_embed, config.dropout)
        
        self.feed_forward5 = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual5 = ResidualConnection(config.d_embed, config.dropout)
        
        # self.feed_forward6 = nn.Sequential(
        #     nn.Linear(config.d_embed, config.d_ff),
        #     nn.GELU(),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(config.d_ff, config.d_embed)
        # )
        # self.residual6 = ResidualConnection(config.d_embed, config.dropout)
        
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
        # print("x after encoder shape", x.shape)
        if pad_mask is not None:
            reshaped_mask = pad_mask.reshape(pad_mask.shape[0], pad_mask.shape[-1], 1)
            reshaped_mask = torch.logical_not(reshaped_mask).float()
            x = x * reshaped_mask
            x = torch.sum(x, dim = 1)
            reshaped_mask_sum = torch.sum(reshaped_mask, dim = 1)
            x_final = x / reshaped_mask_sum
        else:
            x_final = torch.mean(x,-2)
        
        x_final = self.residual1(x_final, self.feed_forward1)
        x_final = self.residual2(x_final, self.feed_forward2)
        x_final = self.residual3(x_final, self.feed_forward3)
        x_final = self.residual4(x_final, self.feed_forward4)
        x_final = self.residual5(x_final, self.feed_forward5)
        # x_final = self.residual6(x_final, self.feed_forward6)
            


        return  self.MLP_head_0(x_final), self.MLP_head_1(x_final)
