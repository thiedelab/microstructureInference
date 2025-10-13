import torch
import torch.nn as nn

### KWANG

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
