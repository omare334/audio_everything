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
# Data loader generator with tokenization
def data_loader(data, batch_size):
    for i in range(0, len(data), batch_size):
        batch_df = data.iloc[i:i+batch_size]
        
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
        
        # Yield the batch
        yield {
            "audio": torch.tensor(np.stack(processed_audio)),
            "input_ids": input_ids,
            "target_ids": target_ids
        }
# data=pd.read_csv(r"J:\common_voice\common\cv-corpus-19.0-2024-09-13\en\train.tsv",sep='\t')
# batch_size = 10
# # Example usage
# for batch in data_loader(data, batch_size):
#     print(batch["audio"].shape)  # Shape of audio Mel spectrograms
#     print(batch["input_ids"].shape)  # Tokenized input sequences
#     print(batch["target_ids"].shape)  # Tokenized target sequences
