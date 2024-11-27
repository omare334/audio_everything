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

base_dir = r"J:\common_voice\common\cv-corpus-19.0-2024-09-13\en\clips"

# Define batch size
batch_size = 128

# Function to process a single batch
def process_batch(batch_df):
    batch_audio = []
    durations = []  # List to store durations of audio in seconds
    for file_name in batch_df['path']:
        # Create the full path
        full_path = os.path.join(base_dir, file_name)
        
        # Load the audio file
        audio, sr = librosa.load(full_path, sr=16000)
        durations.append(len(audio) / sr)  # Calculate duration in seconds
        batch_audio.append(audio)
    
    # Determine the max duration for the batch
    max_time = int(np.ceil(max(durations)))

    
    # Pad or trim each audio sample in the batch to the max duration
    processed_batch = [
        whisper.log_mel_spectrogram(pad_or_trim_custom(audio, max_time)).numpy()
        for audio in batch_audio
    ]
    
    return np.stack(processed_batch)  # Stack into a batch

# Data loader generator
def data_loader(data, batch_size):
    for i in range(0, len(data), batch_size):
        batch_df = data.iloc[i:i+batch_size]  # Get the batch
        yield process_batch(batch_df)

data=pd.read_csv(r"J:\common_voice\common\cv-corpus-19.0-2024-09-13\en\train.tsv",sep='\t')
# Example usage
for batch in data_loader(data, batch_size):
    print(batch.shape)  # Processed batch of Mel spectrograms


