import torch 
import torch.nn as nn
from torch.optim import Adam
from transformers import GPT2Tokenizer
from tqdm import tqdm  # For progress bars
from transformer import Transformer
from data_loader_test import data_loader
import pandas as pd 
import wandb

model_path = r"C:\Users\omare\Desktop\mlx projects\audio_everything\model_cpu_weights.pt"

# Load only weights
state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)