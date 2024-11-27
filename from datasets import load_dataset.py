from datasets import load_dataset

# Print the first 5 audio paths and sentences
# Select only 'audio' and 'sentence' columns
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train[:5]",
                       trust_remote_code=True,streaming=True).select_columns(['audio', 'sentence'])

# Extract the 'audio' and 'sentence' columns
audio_and_sentences = [(entry['audio']['path'], entry['sentence']) for entry in dataset]

# Print the first 5 audio paths and sentences
for audio, sentence in audio_and_sentences[:5]:
    print(f"Audio Path: {audio}, Sentence: {sentence}")
