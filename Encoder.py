import torch
import pandas as pd 
import numpy as np 
import torch.nn.functional as F

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k / denominator)
            P[k, 2*i+1] = np.cos(k / denominator)
    return torch.tensor(P, dtype=torch.float32)

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class Encoder(torch.nn.Module):
    def __init__(self, batch_size=8, n_mel_bins=80, emb_dim = 256, n_time_frames=1000, n_attention_heads=4, hidden_dim=64, n_blocks=4):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.n_mel_bins = n_mel_bins
        self.n_time_frames = n_time_frames  # Attention dimension is half of the time frames
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

        # Conv1D layer to process the mel spectrogram input
        self.conv_layer = conv1d(mel_bins=n_mel_bins,emb_dim=emb_dim, kernel_size=3, padding=1)

        # Positional Encoding
        self.positional_encoding = getPositionEncoding

        # Combined Transformer Blocks (Attention + MLP Block)
        self.transformer_blocks = torch.nn.ModuleList([
            CombinedTransformerBlock(emb_dim=emb_dim, 


                                     num_heads=n_attention_heads, 
                                     hidden_dim=hidden_dim, 
                                     mlp_dim=emb_dim) 
            for _ in range(n_blocks)
        ])

        # Final normalization layer
        self.norm = torch.nn.LayerNorm(emb_dim)

    def forward(self, mel_spectrogram):
        # Conv1D expects input as (batch_size, channels, seq_len), so we permute the input
        # mel_spectrogram: (batch_size, n_mel_bins, n_time_frames)
        conv_output = self.conv_layer(mel_spectrogram)  # (batch_size, n_attention_dim, n_time_frames)

        # Add positional encodings
        batch_size, seq_len, emb_dim = conv_output.size()
        pos_emb = self.positional_encoding(seq_len, emb_dim, 10000).to(conv_output.device)  # Match device
        pos_encoded = conv_output + pos_emb

        # Pass through the sequence of transformer blocks
        output = pos_encoded
        for block in self.transformer_blocks:
            output = block(output)

        # Apply final normalization
        output = self.norm(output)

        return output


class conv1d(torch.nn.Module):
    def __init__(self, mel_bins: int, emb_dim : int, kernel_size: int = 3, padding: int = 1):
        super(conv1d, self).__init__()
        self.norm1 = torch.nn.LayerNorm(mel_bins)  # Normalization before the first convolution
        self.activation = torch.nn.GELU()         # Non-linearity
        self.conv1 = torch.nn.Conv1d(
            in_channels=mel_bins, 
            out_channels=emb_dim, 
            kernel_size=kernel_size,  
            padding=padding
        )
        self.norm2 = torch.nn.LayerNorm(emb_dim)  # Normalization before the second convolution
        self.conv2 = torch.nn.Conv1d(
            in_channels=emb_dim, 
            out_channels=emb_dim, 
            kernel_size=kernel_size, 
            stride= 2, 
            padding=padding
        )
    def forward(self, x):
        """
        Forward pass for mel spectrograms.
        Input shape: (batch_size, n_mel_bins, n_time_frames)
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        return x

class CombinedTransformerBlock(torch.nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, hidden_dim: int, mlp_dim: int):
        super(CombinedTransformerBlock, self).__init__()
        self.attention = MaskedAttention(emb_dim, num_heads)
        self.mlp_block = PreActivationResidualMLPBlock(hidden_dim, mlp_dim)

    def forward(self, x, mask=None):
        x = self.attention(x, None)
        x = self.mlp_block(x)
        return x


    
class MaskedAttention(torch.nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Dimension per head
        # print(self.head_dim)
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        self.linear_q = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_k = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_v = torch.nn.Linear(emb_dim, emb_dim)
        
        self.linear_concat = torch.nn.Linear(emb_dim, emb_dim)

        self.norm = torch.nn.LayerNorm(emb_dim)
        # Learnable bias for attention
        # self.attn_embedding_bias = torch.nn.Parameter(torch.zeros(emb_dim))
        

    def forward(self, emb,mask = None):

        # Fix: Get dimensions correctly using size()
        # seq_len, embed_dim
        batch_size = emb.size(0)
        seq_len = emb.size(1)
        
        # Transform embeddings for query, key, and value
        query = self.linear_q(emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores and apply softmax
        scaling_factor = self.head_dim ** 0.5
        similarity_matrix = (query @ key.transpose(-2, -1)) / scaling_factor

        if mask is not None:
            mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1) * -1e9
            similarity_matrix = similarity_matrix + mask
            return similarity_matrix

        # Apply softmax to get attention weights
        soft_matrix = torch.softmax(similarity_matrix, dim=-1)
    
        # Apply attention weights to values and reshape back
        attention = torch.matmul(soft_matrix, value)
        attention = attention.transpose(1, 2).contiguous()
        attn_emb = attention.view(batch_size,seq_len, -1)  # Reshape

        attn_emb = self.linear_concat(attn_emb)

        attn_emb = self.norm(attn_emb + emb)

        # add residual and normalisation
        
        return attn_emb

class PreActivationResidualMLPBlock(torch.nn.Module):
    # This is the MLP mixed with the preactivation t
    def __init__(self, hidden_dim: int, mlp_dim: int):
        super(PreActivationResidualMLPBlock, self).__init__()
        
        self.norm1 = torch.nn.LayerNorm(mlp_dim)  
        self.activation = torch.nn.GELU()           
        self.linear1 = torch.nn.Linear(mlp_dim, hidden_dim)  
        self.norm2 = torch.nn.LayerNorm(hidden_dim)    
        self.linear2 = torch.nn.Linear(hidden_dim, mlp_dim)  
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.linear1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.linear2(x)
        x += residual
        return x
