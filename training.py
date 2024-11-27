import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import GPT2Tokenizer
from tqdm import tqdm  # For progress bars
from transformer import Transformer
from data_loader_test import data_loader
import pandas as pd 

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = "<pad>"

# Hyperparameters for Encoder and Decoder
encoder_config = {
    'batch_size': 8,
    'n_mel_bins': 80,
    'n_attention_heads': 2,
    'hidden_dim': 64,
    'n_blocks': 12
}

decoder_config = {
    'batch_size': 8,
    'Wemb_dim': 768,
    'Pemb_dim': 500,  # Dynamically adjusted during training
    'num_heads': 2,
    'hidden_dim': 64,
    'mlp_dim': 128,
    'n_blocks': 12,
    'voc_size': 50264  # Vocabulary size from tokenizer
}

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(encoder_config, decoder_config).to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
def train_model(data_loader, model, optimizer, criterion, epochs=1):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Retrieve batched data
            audio = batch["audio"].to(device)  # Mel spectrograms
            input_ids = batch["input_ids"].to(device)  # Input tokens
            target_ids = batch["target_ids"].to(device)  # Target tokens
            
            # Update Pemb_dim dynamically based on input audio
            n_time_frames = audio.size(-1)
            decoder_config['Pemb_dim'] = n_time_frames // 2

            # Forward pass
            outputs = model(audio, input_ids)

            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_len, voc_size)
            targets = target_ids.view(-1)  # (batch_size * seq_len)

            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Prepare data loader
# Replace `data_loader` with your actual data loader generator
data = pd.read_csv(r"J:\common_voice\common\cv-corpus-19.0-2024-09-13\en\train.tsv", sep='\t')
batch_size = 8
data_loader = data_loader(data, batch_size)

# Train the model
train_model(data_loader, model, optimizer, criterion, epochs=5)
