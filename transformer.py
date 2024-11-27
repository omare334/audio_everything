import torch 
from Encoder import Encoder
from Decoder import Decoder

class Transformer(torch.nn.Module):
    def __init__(self, 
                 encoder_config: dict,
                 decoder_config: dict):
        """
        Combined Transformer class consisting of an Encoder and a Decoder.

        Args:
        - encoder_config (dict): Configuration dictionary for the Encoder.
        - decoder_config (dict): Configuration dictionary for the Decoder.
        """
        super(Transformer, self).__init__()

        # Initialize Encoder
        self.encoder = Encoder(**encoder_config)

        # Initialize Decoder
        self.decoder = Decoder(**decoder_config)

    def forward(self, mel_spectrogram, tokens):
        """
        Forward pass for the Transformer.

        Args:
        - mel_spectrogram (torch.Tensor): Input mel spectrogram for the Encoder.
        - tokens (torch.Tensor): Tokenized input for the Decoder.

        Returns:
        - torch.Tensor: Final output from the Decoder.
        """
        # Pass through Encoder
        patches = self.encoder(mel_spectrogram)  # Output of the Encoder

        # Pass through Decoder
        output = self.decoder(tokens, patches)

        return output

if __name__ == "__main__":
    # Define hyperparameters for Encoder and Decoder
    encoder_config = {
        'batch_size': 8,
        'n_mel_bins': 80,
        'n_time_frames': 3000,
        'n_attention_heads': 2,
        'hidden_dim': 64,
        'n_blocks': 12
    }

    decoder_config = {
        'batch_size': 8,
        'Wemb_dim': 768,
        'Pemb_dim': 1500,
        'num_heads': 2,
        'hidden_dim': 64,
        'mlp_dim': 128,
        'n_blocks': 12,
        'voc_size': 50300
    }

    # Initialize the Transformer model
    model = Transformer(encoder_config, decoder_config)

    # Generate random inputs
    # Mel spectrogram input for Encoder: (batch_size, n_mel_bins, n_time_frames)
    mel_spectrogram = torch.randn(encoder_config['batch_size'], 
                                  encoder_config['n_mel_bins'], 
                                  encoder_config['n_time_frames'])

    # Token input for Decoder: (batch_size, seq_len)
    seq_len = 38  # Sequence length for tokens
    tokens = torch.randint(0, decoder_config['voc_size'], 
                           (decoder_config['batch_size'], seq_len))

    # Move model and inputs to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mel_spectrogram = mel_spectrogram.to(device)
    tokens = tokens.to(device)

    # Forward pass
    output = model(mel_spectrogram,tokens)
    print("Transformer output shape:", output.shape)
