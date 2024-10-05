import os
import json
import random
from pathlib import Path
import warnings

# ============================
# Set NCCL environment variables
# ============================
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Define file paths
DATA_DIR = 'data'
CHALLENGES_FILE = os.path.join(DATA_DIR, 'arc-agi_training_challenges.json')
CODES_FILE = os.path.join(DATA_DIR, 'arc_training_codes.json')

# Load JSON data
with open(CHALLENGES_FILE, 'r') as f:
    challenges = json.load(f)

with open(CODES_FILE, 'r') as f:
    codes = json.load(f)

# Define separator token
SEPARATOR = "\n===\n"

# Prepare dataset entries
dataset_entries = []

for key, code in codes.items():
    if key not in challenges:
        continue  # Skip if no corresponding challenge

    challenge = challenges[key]
    train_examples = challenge.get('train', [])

    # Skip if no training examples
    if not train_examples:
        continue

    # Create prompt by concatenating training input-output grids
    prompt_parts = []
    for example in train_examples:
        input_grid = example.get('input', [])
        output_grid = example.get('output', [])
        input_str = json.dumps(input_grid)
        output_str = json.dumps(output_grid)
        prompt_parts.append(f"Input Grid: {input_str}\nOutput Grid: {output_str}")

    prompt = f"Training Examples:\n" + f"{SEPARATOR}".join(prompt_parts) + "\n\nCode Completion:\n"

    # Append to dataset entries
    dataset_entries.append({
        'prompt': prompt,
        'completion': code
    })

# Limit to 400 entries
if len(dataset_entries) > 400:
    dataset_entries = random.sample(dataset_entries, 400)

# Split into training and validation sets (95% train, 5% validation)
train_size = int(0.95 * len(dataset_entries))
val_size = len(dataset_entries) - train_size
train_entries, val_entries = random_split(dataset_entries, [train_size, val_size])

class ARCCodeDataset(Dataset):
    def __init__(self, entries, tokenizer, max_length=900):
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        prompt = entry['prompt']
        completion = entry['completion']

        # Concatenate prompt and completion
        text = prompt + completion

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Labels are the same as input_ids, but we mask the prompt part
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]

        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Ignore the prompt in loss

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class PrintSampleCallback(TrainerCallback):
    """
    A custom callback that prints a decoded validation prediction after each epoch.
    """
    def __init__(self, tokenizer, val_dataset, max_new_tokens=100, num_beams=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def on_epoch_end(self, args, state, control, **kwargs):
        # Select a random sample from the validation dataset
        sample = random.choice(self.val_dataset)
        prompt_ids = sample['input_ids'].unsqueeze(0).to(kwargs['model'].device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(kwargs['model'].device)

        # Generate prediction
        with torch.no_grad():
            output_ids = kwargs['model'].generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,  # Use max_new_tokens instead of max_length
                num_beams=self.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id  # Ensure pad_token_id is set
            )

        # Decode the generated tokens, skip the prompt
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
        completion_text = generated_text[len(prompt_text):].strip()

        print("\n--- Sample Validation Prediction ---")
        print(f"Prompt:\n{prompt_text}")
        print(f"Generated Completion:\n{completion_text}")
        print("-----------------------------------\n")

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# ============================
# Set the pad_token to eos_token
# ============================
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model.resize_token_embeddings(len(tokenizer))  # Important if pad_token is added

# Create datasets
train_dataset = ARCCodeDataset(train_entries, tokenizer)
val_dataset = ARCCodeDataset(val_entries, tokenizer)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments with updated `eval_strategy` and `ddp_find_unused_parameters`
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=100,
    save_steps=500,
    eval_strategy='epoch',  # Updated from 'evaluation_strategy'
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    ddp_find_unused_parameters=False,  # Set to False to resolve DDP warning
    # Additional arguments to optimize training on a single GPU
    # Uncomment the following line if you're training on CPU
    # no_cuda=True,
)

# Initialize custom callback
print_callback = PrintSampleCallback(
    tokenizer=tokenizer,
    val_dataset=val_dataset,
    max_new_tokens=100,  # Specify how many new tokens to generate
    num_beams=5
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[print_callback],  # Add the custom callback here
)

# Train the model
trainer.train()

# Evaluate on validation set
eval_results = trainer.evaluate()

print(f"Validation Loss: {eval_results['eval_loss']}")

