from transformers import GPT2Tokenizer
import numpy as np
import torch
import whisper
import matplotlib.pyplot as plt
from datasets import load_dataset

def pad_or_trim_custom(audio, max_time=5, sr=16000):
    """
    Pads or trims the audio to the specified target length.

    Args:
        audio (torch.Tensor): Input audio tensor.
        max_time (int): Desired length in seconds (default is 5 seconds).
        sr (int): Sampling rate (default is 16kHz).
        
    Returns:
        torch.Tensor: Audio tensor of length `target_length`.
    """
    target_length = max_time * sr
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        padding = target_length - len(audio)
        return torch.nn.functional.pad(audio, (0, padding), mode='constant', value=0)
    else:
        return audio

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add special tokens
special_tokens = ["<pad>", "<start>", "<end>", "<transcribe>", "<translate>", "<en>", "<ar>"]
tokenizer.add_tokens(special_tokens)

# Set the special tokens
tokenizer.pad_token = "<pad>"

# Load the dataset
dataset = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train", streaming=True)

# Data loader generator with tokenization
def data_loader(dataset, batch_size):
    batch_audio = []
    durations = []  # Store audio durations
    tokenized_inputs = []  # Tokenized input sequences
    tokenized_targets = []  # Tokenized target sequences
    
    batch = []
    for row in dataset:
        # Load and process the audio from the dataset
        audio_array = torch.tensor(row['audio']['array'], dtype=torch.float32)  # Convert to tensor
        sr = row['audio']['sampling_rate']
        durations.append(len(audio_array) / sr)
        batch_audio.append(audio_array)
        
        # Determine the task token
        task_token = "<translate>" if row["locale"] in ["en"] else "<transcribe>"
        
        # Build the structured input
        input_text = f"<start>{row['locale']}{task_token} {row['sentence']}"
        target_text = f"{row['locale']}{task_token} {row['sentence']}<end>"
        
        # Tokenize the input and target
        tokenized_input = tokenizer.encode(input_text, add_special_tokens=False)
        tokenized_target = tokenizer.encode(target_text, add_special_tokens=False)
        
        tokenized_inputs.append(tokenized_input)
        tokenized_targets.append(tokenized_target)
        
        # Add row to the batch
        batch.append(row)
        
        # Yield a batch when the size reaches batch_size
        if len(batch) == batch_size:
            # Pad or trim audio and tokens
            max_time = int(np.ceil(max(durations)))
            processed_audio = [
                whisper.log_mel_spectrogram(pad_or_trim_custom(audio, 10)).numpy()
                for audio in batch_audio
            ]
            
            # Pad tokenized sequences
            input_ids = tokenizer.pad(
                {"input_ids": tokenized_inputs},
                padding=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )["input_ids"]
            
            target_ids = tokenizer.pad(
                {"input_ids": tokenized_targets},
                padding=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )["input_ids"]
            
            yield {
                "audio": torch.tensor(np.stack(processed_audio), dtype=torch.float32),
                "input_ids": input_ids,
                "target_ids": target_ids
            }
            
            # Reset batch and other variables for the next batch
            batch_audio = []
            durations = []
            tokenized_inputs = []
            tokenized_targets = []
            batch = []

# Plotting function for Mel spectrogram
def plot_mel_spectrogram(mel_spectrogram, title="Mel Spectrogram"):
    """
    Plots a Mel spectrogram.

    Args:
        mel_spectrogram (np.ndarray): The Mel spectrogram to plot.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()

# Example usage
batch_size = 64
num_batches_to_test = 5  # Specify the number of batches to test
batch_count = 0  # Counter to track the number of processed batches

for batch in data_loader(dataset, batch_size=batch_size):
    print(f"Batch {batch_count + 1}:")
    print("Audio shape:", batch["audio"].shape)
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Target IDs shape:", batch["target_ids"].shape)
    print("-" * 50)  # Separator for readability
    
    batch_count += 1
    if batch_count == num_batches_to_test:
        break  # Stop after processing the specified number of batches

