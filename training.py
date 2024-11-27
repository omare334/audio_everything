import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import GPT2Tokenizer
from tqdm import tqdm  # For progress bars
from transformer import Transformer
from data_loader_test import data_loader
import pandas as pd 
import wandb

# Hyperparameters for Encoder and Decoder
encoder_config = {
    'batch_size': 64,
    'n_mel_bins': 80,
    'n_attention_heads': 2,
    'hidden_dim': 64,
    'n_blocks': 2
}

decoder_config = {
    'batch_size': 64,
    'Wemb_dim': 768,
    'Pemb_dim': 500,  # Dynamically adjusted during training
    'num_heads': 2,
    'hidden_dim': 64,
    'mlp_dim': 128,
    'n_blocks': 2,
    'voc_size': 50264  # Vocabulary size from tokenizer
}

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Add special tokens
special_tokens = ["<pad>", "<start>", "<end>", "<transcribe>", "<translate>", "<en>", "<ar>"]
tokenizer.add_tokens(special_tokens)

tokenizer.pad_token = "<pad>"

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(encoder_config, decoder_config).to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Initialize W&B
wandb.init(project="audio-to-text-transformer")

# Define the number of epochs
num_epochs = 10

# Load your training data
data_path = r"J:\common_voice\common\cv-corpus-19.0-2024-09-13\en\train.tsv"
data = pd.read_csv(data_path, sep='\t')
data = data[:10000]

# Create the data loader
batch_size = encoder_config["batch_size"]
train_loader = data_loader(data, batch_size)

# Move model to device and set it to training mode
model.train()

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    batch_size = 64
    num_batches = len(data) // batch_size  # Calculate number of batches
    
    # Create the data loader for this epoch
    train_loader = data_loader(data, batch_size)
    
    # Create progress bar based on number of batches
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=num_batches, unit="batch")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        audio = batch["audio"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(audio, input_ids)
        
        # Compute loss
        outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_len, voc_size)
        targets = target_ids.view(-1)  # (batch_size * seq_len)
        
        # Compute loss
        loss = criterion(outputs, targets)
        epoch_loss += loss.item()  # Add batch loss to epoch loss
        
        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()
        
        # Update W&B
        wandb.log({
            "loss": loss.item(),
            "epoch": epoch + 1,
            "batch_idx": batch_idx + 1,
        })
        
        # Update the progress bar with the loss
        pbar.set_postfix(loss=loss.item())
    
    # Log average epoch loss to W&B
    epoch_avg_loss = epoch_loss / num_batches  # Average loss for the entire epoch
    wandb.log({"epoch_loss": epoch_avg_loss, "epoch": epoch + 1})
    print(f"Epoch {epoch + 1} Loss: {epoch_avg_loss}")

# Finish the W&B logging
wandb.finish()

# Save the trained model to W&B
torch.save(model.state_dict(), "transformer_model.pt")

