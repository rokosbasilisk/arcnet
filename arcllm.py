import os
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set the EOS token as the padding token

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

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
train_size = int(0.8 * len(dataset_entries))
val_size = len(dataset_entries) - train_size
train_entries, val_entries = random_split(dataset_entries, [train_size, val_size])

class ARCCodeDataset(Dataset):
    def __init__(self, entries, tokenizer, max_length=1024):
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
        prompt_encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors='pt')
        prompt_length = prompt_encoding['input_ids'].shape[1]

        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Ignore the prompt in loss

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Initialize model
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model.resize_token_embeddings(len(tokenizer))

# Create datasets
train_dataset = ARCCodeDataset(train_entries, tokenizer)
val_dataset = ARCCodeDataset(val_entries, tokenizer)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=100,
    save_steps=500,
    evaluation_strategy='epoch',
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Function to generate a completion for a random validation sample
# Function to generate a completion for a random validation sample
def generate_random_completion():
    model.eval()  # Set model to evaluation mode
    random_index = random.randint(0, len(val_dataset) - 1)
    val_sample = val_dataset[random_index]
    
    # Decode the prompt
    input_ids = val_sample['input_ids'].unsqueeze(0).to(model.device)
    prompt_length = (val_sample['labels'] == -100).sum().item()  # Count masked tokens
    prompt_ids = input_ids[:, :prompt_length]

    prompt_text = tokenizer.decode(prompt_ids.squeeze(), skip_special_tokens=True)

    # Ensure that the total length (input + generated) does not exceed 1024
    max_length = 1024
    max_new_tokens = max_length - prompt_length  # Adjust new tokens based on the input length

    # Cap the number of new tokens to ensure we stay within the limit
    max_new_tokens = min(max_new_tokens, 200)  # Set a reasonable limit for new tokens

    if max_new_tokens <= 0:
        print("\n--- Random Validation Example ---")
        print("Prompt:\n", prompt_text)
        print("No room for completion generation, prompt is too long.")
        return

    # Generate completion with the adjusted number of tokens
    output_ids = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=max_new_tokens,  # Generate tokens while ensuring total length < 1024
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n--- Random Validation Example ---")
    print("Prompt:\n", prompt_text)
    print("Completion:\n", output_text)

# Training loop with custom logging after each epoch
for epoch in range(training_args.num_train_epochs):
    print(f"\nStarting epoch {epoch+1}/{training_args.num_train_epochs}...")
    trainer.train()  # Train for one epoch

    # Evaluate on validation set
    eval_results = trainer.evaluate()
    print(f"Epoch {epoch+1} Validation Loss: {eval_results['eval_loss']}")

    # Print a random validation completion
    generate_random_completion()

# Save the model once the training is completed
trainer.save_model("output_model")

