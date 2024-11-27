from Encoder import MaskedAttention,PreActivationResidualMLPBlock,getPositionEncoding,conv1d
import torch
from transformers import GPT2Config, GPT2Model
from transformers import GPT2Tokenizer, GPT2LMHeadModel
custom_config = GPT2Config(
    vocab_size=50300,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_head=12,
    n_layer=12,
    resid_pdrop=0.1,
    attn_pdrop=0.1,
    embd_pdrop=0.1,
    use_cache=True
)

import torch
import torch.nn as nn

class DecoderPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, positional_embedding_size):
        super(DecoderPositionalEmbedding, self).__init__()
        
        # Positional embedding of size `positional_embedding_size`
        self.positional_embedding = nn.Embedding(max_seq_len, positional_embedding_size)

    def forward(self, seq_len, batch_size):
        """
        Args:
            seq_len: int, length of the current sequence.
            batch_size: int, size of the batch.
        
        Returns:
            pos_embeds: Tensor of shape (batch_size, seq_len, positional_embedding_size).
        """
        # Generate position indices (0, 1, 2, ..., seq_len-1)
        positions = torch.arange(seq_len, device=self.positional_embedding.weight.device)
        # Expand positions for the entire batch
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)  # Shape: (batch_size, seq_len)
        # Get positional embeddings
        pos_embeds = self.positional_embedding(positions)  # Shape: (batch_size, seq_len, positional_embedding_size)
        return pos_embeds


class Decoder(torch.nn.Module):
    def __init__(self, batch_size=8, Wemb_dim = 768, Pemb_dim = 64, num_heads = 4, hidden_dim = 64, mlp_dim = 64,n_blocks =4,voc_size=53000):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2.resize_token_embeddings(len(voc_size))
        # Positional Encoding
        self.positional_encoding = getPositionEncoding
        # Combined Transformer Blocks (Attention + MLP Block)
        self.transformer_blocks = torch.nn.ModuleList([
            CombinedTransformerBlock(Wemb_dim,Pemb_dim,
                                      num_heads, hidden_dim,
                                        mlp_dim, voc_size = voc_size) 
            for _ in range(n_blocks)
        ])
        # Final normalization layer
        self.norm = torch.nn.LayerNorm(Wemb_dim)
        self.positional_embedding = DecoderPositionalEmbedding(200, 768)
        self.project = torch.nn.Linear(Wemb_dim, voc_size)


    def forward(self, tokens,patches):
        word_embeddings = self.gpt2.transformer.wte(tokens)
        batch_size, seq_len, emb_dim = word_embeddings.size()
        pos_emb = self.positional_embedding(seq_len, batch_size)
        pos_encoded = word_embeddings + pos_emb
        output = pos_encoded
        for block in self.transformer_blocks:
            output = block(output,patches)
        output = self.norm(output)
        output = self.project(output)
        return output

class CombinedTransformerBlock(torch.nn.Module):
    def __init__(self, Wemb_dim: int,Pemb_dim :int, num_heads: int, hidden_dim: int, mlp_dim: int,voc_size:int):
        super(CombinedTransformerBlock, self).__init__()
        self.attention = MaskedAttention(Wemb_dim, num_heads)
        self.cross_attention = CrossAttention(Wemb_dim, Pemb_dim, mlp_dim, num_heads, voc_size)
        self.mlp_block = PreActivationResidualMLPBlock(hidden_dim, Wemb_dim)

    def forward(self, tokens,patches, mask=True):
        x = self.attention(tokens, mask)
        x = self.cross_attention(tokens,patches)
        x = self.mlp_block(x)
        return x
    
class CrossAttention(torch.nn.Module):
    def __init__(self, Wemb_dim, Pemb_dim, new_dim, num_heads, voc_size):
        super().__init__()
        self.num_heads = num_heads
        self.Whead_dim = new_dim // num_heads  # Embedding dimension for words per head
        self.Phead_dim = new_dim // num_heads  # Embedding dimension for images per head

        assert Wemb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        assert Pemb_dim % num_heads == 0, "Embedding dimension for images must be divisible by the number of heads"

        self.embeddings = torch.nn.Embedding(num_embeddings=voc_size, embedding_dim=Wemb_dim)

        # Linear layers for query, key, and value transformations
        self.linear_q = torch.nn.Linear(Wemb_dim, new_dim)
        self.linear_k = torch.nn.Linear(Pemb_dim, new_dim)
        self.linear_v = torch.nn.Linear(Pemb_dim, new_dim)

        # Linear layer for the concatenated output
        self.linear_concat = torch.nn.Linear(new_dim, Wemb_dim)

        self.norm = torch.nn.LayerNorm(Wemb_dim)

    def forward(self, wemb, pemb):
        # wemb: [batch_size, seq_len_w, Wemb_dim]
        # pemb: [batch_size, seq_len_p, Pemb_dim]
        # batch_size,seq_len, wemb_dim
        # No positional encoding needed for image embeddings (Pemb)
        batch_size = wemb.size(0)
        batch_size = pemb.size(0)
        Wseq_len = wemb.size(1)
        Pseq_len = pemb.size(1)
        
     
        print("The Pemb shape:", pemb.shape) 

        # Transform embeddings for query, key, and value
        query = self.linear_q(wemb).view(batch_size, Wseq_len, self.num_heads, self.Whead_dim).transpose(1, 2)
        key = self.linear_k(pemb).view(batch_size, Pseq_len, self.num_heads, self.Phead_dim).transpose(1, 2)
        value = self.linear_v(pemb).view(batch_size, Pseq_len, self.num_heads, self.Phead_dim).transpose(1, 2)

        print("Query shape after linear transformation:", query.shape)
        print("Key shape after linear transformation:", key.shape)
        print("Value shape after linear transformation:", value.shape)

        # Attention computation: query * key^T
        scaling_factor = self.Whead_dim ** 0.5  # or use self.Phead_dim if necessary
        attention = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

        # Apply softmax to get attention weights
        soft_matrix = torch.softmax(attention, dim=-1)

        # Attention output
        # matmul replacement
        sim_mat = soft_matrix @ value  
        sim_mat = sim_mat.transpose(1, 2).contiguous()
        final_emb = sim_mat.view(batch_size, Wseq_len, -1)  # Reshape to (Wseq_len, num_heads * Whead_dim)

        # Pass through the linear layer after concatenation
        final_emb = self.linear_concat(final_emb)

        final_emb = self.norm(final_emb + wemb)
        # add residual and normalization

        return final_emb
    
    import torch

if __name__ == "__main__":
    # Define hyperparameters
    batch_size = 8
    seq_len = 32  # Sequence length for tokens
    patch_seq_len = 16  # Sequence length for patches
    Wemb_dim = 768
    Pemb_dim = 64
    voc_size = 5297

    # Initialize the Decoder model
    model = Decoder(
        batch_size=batch_size,
        Wemb_dim=Wemb_dim,
        Pemb_dim=Pemb_dim,
        num_heads=4,
        hidden_dim=64,
        mlp_dim=128,  # Adjust as necessary
        n_blocks=4,
        voc_size=voc_size,
    )

    # Generate random token inputs (e.g., vocabulary indices)
    tokens = torch.randint(0, voc_size, (batch_size, seq_len))
    # Generate random patch inputs (e.g., patch embeddings)
    patches = torch.randn(batch_size, patch_seq_len, Pemb_dim)

    # Move model and inputs to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokens = tokens.to(device)
    patches = patches.to(device)

    # Forward pass

    output = model(tokens, patches)
    print("Model output shape:", output.shape)
