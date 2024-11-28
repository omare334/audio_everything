from transformers import GPT2Tokenizer
import pandas as pd 
import numpy as np 
import torch 
import whisper
import librosa
import matplotlib.pyplot as plt
import os 

def pad_or_trim_custom(audio, max_time=5,sr = 16000):
    """
    Pads or trims the audio to the specified target length.
    
    Args:
        audio (np.ndarray): Input audio array.
        target_length (int): Desired length in samples (default is 80,000 for 5 seconds at 16kHz).
        
    Returns:
        np.ndarray: Audio array of length `target_length`.
    """
    target_length = max_time * 16000
    if len(audio) > target_length:
        # Trim the audio to the target length
        return audio[:target_length]
    elif len(audio) < target_length:
        # Pad with zeros at the end to match the target length
        padding = target_length - len(audio)
        return np.pad(audio, (0, padding), mode='constant')
    else:
        # Audio is already the correct length
        return audio
# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add special tokens
special_tokens = ["<pad>", "<start>", "<end>", "<transcribe>", "<translate>", "<en>", "<ar>"]
tokenizer.add_tokens(special_tokens)

# Set the special tokens
tokenizer.pad_token = "<pad>"
base_dir = r"J:\common_voice\common\cv-corpus-19.0-2024-09-13\en\clips"


def get_batch_by_index(data, batch_size, index):
    """
    Retrieves a specific batch from the dataset based on the given index.
    
    Args:
        data (pd.DataFrame): The dataset to retrieve batches from.
        batch_size (int): The size of each batch.
        index (int): The index of the batch to retrieve.
        
    Returns:
        dict: A dictionary containing processed audio, input_ids, and target_ids for the batch.
    """
    start_idx = index * batch_size
    end_idx = min(start_idx + batch_size, len(data))
    
    if start_idx >= len(data):
        raise IndexError("Batch index out of range.")
    
    batch_df = data.iloc[start_idx:end_idx]
    
    batch_audio = []
    durations = []  # Store audio durations
    tokenized_inputs = []  # Tokenized input sequences
    tokenized_targets = []  # Tokenized target sequences
    
    for _, row in batch_df.iterrows():
        # Load and process the audio
        full_path = os.path.join(base_dir, row['path'])
        audio, sr = librosa.load(full_path, sr=16000)
        durations.append(len(audio) / sr)
        batch_audio.append(audio)
        
        # Determine the task token
        task_token = "<transcribe>" if row["locale"] in ["en"] else "<translate>"
        
        # Build the structured input
        input_text = f"<start>{row['locale']}{task_token} {row['sentence']}"
        target_text = f"{row['locale']}{task_token} {row['sentence']}<end>"
        
        # Tokenize the input and target
        tokenized_input = tokenizer.encode(input_text, add_special_tokens=False)
        tokenized_target = tokenizer.encode(target_text, add_special_tokens=False)
        
        tokenized_inputs.append(tokenized_input)
        tokenized_targets.append(tokenized_target)
    
    # Pad or trim audio and tokens
    max_time = int(np.ceil(max(durations)))
    processed_audio = [
        whisper.log_mel_spectrogram(pad_or_trim_custom(audio, max_time)).numpy()
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
    
    return {
        "audio": torch.tensor(np.stack(processed_audio)),
        "input_ids": input_ids,
        "target_ids": target_ids
    }

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

if __name__ == "__main__":
    data = pd.read_csv(r"J:\common_voice\common\cv-corpus-19.0-2024-09-13\en\train.tsv", sep='\t')
    batch_size = 64
    
    # Get batch 1278
    batch_index = 1278

    
    try:
        batch = get_batch_by_index(data, batch_size, batch_index)
        print(f"Batch Index: {batch_index}")
        print(f"Audio Shape: {batch['audio'].shape}")  # Shape of audio Mel spectrograms
        print(f"Input IDs Shape: {batch['input_ids'].shape}")  # Shape of tokenized input sequences
        print(f"Target IDs Shape: {batch['target_ids'].shape}")  # Shape of tokenized target sequences
        
        # Visualize a Mel spectrogram from the batch
        mel_spectrogram = batch["audio"][0].numpy()  # Extract the first spectrogram
        plot_mel_spectrogram(mel_spectrogram, title=f"Mel Spectrogram - Batch {batch_index}")
    except IndexError as e:
        print(e)
