from Encoder import conv1d,MaskedAttention,PreActivationResidualMLPBlock,Encoder
import torch 


# if __name__ == "__main__":
#     # Parameters for the mel spectrogram
#     batch_size = 8
#     n_mel_bins = 80
#     n_time_frames = 1000

#     # Create a dummy mel spectrogram tensor
#     mel_spectrogram = torch.randn(batch_size, n_mel_bins, n_time_frames)

#     # Initialize the Conv1D module
#     conv_layer = conv1d(channels=n_mel_bins, kernel_size=3, padding=1)

#     # Pass the mel spectrogram through the Conv1D module
#     output_conv = conv_layer(mel_spectrogram)

#     attention = MaskedAttention(500,4)

#     attnetion_output = attention(output_conv)

#     mlp = PreActivationResidualMLPBlock(64,500)

#     mlp_output = mlp(attnetion_output)
#     # Print input and output shapes
#     print(f"Input shape: {mel_spectrogram.shape}")

    # Example usage:
# if __name__ == "__main__":
#     # Parameters for the mel spectrogram
#     batch_size = 8
#     n_mel_bins = 80
#     n_time_frames = 1000

#     # Create a dummy mel spectrogram tensor
#     mel_spectrogram = torch.randn(batch_size, n_mel_bins, n_time_frames)

#     # Initialize the model
#     model = Encoder()

#     model_output = model(mel_spectrogram)

#     print(model_output.shape)

import torch
import torch.nn as nn

class DecoderWithWeightTying(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super(DecoderWithWeightTying, self).__init__()
        self.embed_dim = embed_dim

        # Word embedding (shared for input and output)
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embedding
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Output projection layer (tied to the word_embedding weights)
        self.output_projection = nn.Linear(embed_dim, vocab_size)

        # Tie the weights of the output projection to the word embedding
        self.output_projection.weight = self.word_embedding.weight

    def forward(self, input_ids):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len), token indices for input sequence.
        
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size), output logits for prediction.
        """
        batch_size, seq_len = input_ids.size()

        # Create positional indices (0, 1, 2, ..., seq_len-1) for each sequence
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        # Get word and positional embeddings
        word_embeds = self.word_embedding(input_ids)  # Shape: (batch_size, seq_len, embed_dim)
        position_embeds = self.positional_embedding(positions)  # Shape: (batch_size, seq_len, embed_dim)

        # Combine word and positional embeddings
        embeddings = word_embeds + position_embeds  # Shape: (batch_size, seq_len, embed_dim)

        # Compute output logits using the tied weight projection
        logits = self.output_projection(embeddings)  # Shape: (batch_size, seq_len, vocab_size)

        return logits

# Example usage:
vocab_size = 10000
embed_dim = 512
max_seq_len = 100

decoder = DecoderWithWeightTying(vocab_size, embed_dim, max_seq_len)

# Sample input IDs (batch of tokenized sequences)
input_ids = torch.randint(0, vocab_size, (16, 20))  # Batch of 16 sequences, each of length 20
logits = decoder(input_ids)

print(logits.shape)  # Should be (16, 20, 10000)

