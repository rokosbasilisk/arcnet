import os
import json
import random
import numpy as np
import warnings
from pathlib import Path

# ============================
# Set NCCL environment variables
# ============================
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# Removed CUDA_VISIBLE_DEVICES to allow Trainer to utilize all available GPUs
model_name = "meta-llama/Llama-3.2-3B"  # Using the 3B model as requested

import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

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
FUNCTIONS_CONTEXT_FILE = os.path.join(DATA_DIR, 'functions_context.json')  # Path to the functions context

# Load JSON data
with open(CHALLENGES_FILE, 'r') as f:
    challenges = json.load(f)

with open(CODES_FILE, 'r') as f:
    codes = json.load(f)

# Load Functions Context
with open(FUNCTIONS_CONTEXT_FILE, 'r') as f:
    functions_context = json.load(f)

# Convert functions_context JSON to a formatted string for inclusion in prompts
functions_context_str = "## Function Definitions\n\n"
for func in functions_context:
    args = ", ".join(func["arguments"])
    functions_context_str += f"**{func['name']}({args}) -> {func['return_type']}**: {func['description']}\n\n"

# Define grid compression and decompression functions
def compress_grid(grid):
    flattened = [str(cell) for row in grid for cell in row]
    compressed = []
    current_char = flattened[0]
    count = 1

    for char in flattened[1:]:
        if char == current_char:
            count += 1
        else:
            compressed.append(f"{current_char}{count}")
            current_char = char
            count = 1

    compressed.append(f"{current_char}{count}")
    return "".join(compressed)

def decompress_grid(compressed, rows, cols):
    decompressed = []
    i = 0

    while i < len(compressed):
        char = compressed[i]
        i += 1
        count = ''
        while i < len(compressed) and compressed[i].isdigit():
            count += compressed[i]
            i += 1
        decompressed.extend([int(char)] * int(count))
    
    grid = [decompressed[i:i + cols] for i in range(0, len(decompressed), cols)]
    return grid

# Define separator token
SEPARATOR = "\n===\n"

# Prepare dataset entries
dataset_entries = []

# Prepare context for model (including function definitions)
context = f"""
{functions_context_str}
The following functions are used for transforming grids:

def compress_grid(grid):
    # Converts a 2D grid into a compressed string representation.

    flattened = [str(cell) for row in grid for cell in row]
    compressed = []
    current_char = flattened[0]
    count = 1

    for char in flattened[1:]:
        if char == current_char:
            count += 1
        else:
            compressed.append(f"{current_char}{count}")
            current_char = char
            count = 1

    compressed.append(f"{current_char}{count}")
    return "".join(compressed)

def decompress_grid(compressed, rows, cols):
    # Converts a compressed string back into a 2D grid

    decompressed = []
    i = 0

    while i < len(compressed):
        char = compressed[i]
        i += 1
        count = ''
        while i < len(compressed) and compressed[i].isdigit():
            count += compressed[i]
            i += 1
        decompressed.extend([int(char)] * int(count))

    grid = [decompressed[i:i + cols] for i in range(0, len(decompressed), cols)]
    return grid

"""

for key, code in codes.items():
    if key not in challenges:
        continue  # Skip if no corresponding challenge

    challenge = challenges[key]
    train_examples = challenge.get('train', [])

    # Skip if no training examples
    if not train_examples:
        continue

    # Create prompt by compressing training input-output grids
    prompt_parts = []
    for example in train_examples:
        input_grid = example.get('input', [])
        output_grid = example.get('output', [])
        compressed_input = compress_grid(input_grid)
        compressed_output = compress_grid(output_grid)
        prompt_parts.append(f"Compressed Input: {compressed_input}\nCompressed Output: {compressed_output}")

    prompt = f"{context}\nTraining Examples:\n" + f"{SEPARATOR}".join(prompt_parts) + "\n\nCode Completion:\n"

    # Append to dataset entries
    dataset_entries.append({
        'prompt': prompt,
        'completion': code
    })

# Measure prompt and completion lengths
prompt_lengths = [len(entry['prompt']) for entry in dataset_entries]
completion_lengths = [len(entry['completion']) for entry in dataset_entries]

max_prompt_length = max(prompt_lengths)
median_prompt_length = np.median(prompt_lengths)
max_completion_length = max(completion_lengths)
median_completion_length = np.median(completion_lengths)

print(f"Maximum Prompt Length: {max_prompt_length} characters")
print(f"Median Prompt Length: {median_prompt_length} characters")
print(f"Maximum Completion Length: {max_completion_length} characters")
print(f"Median Completion Length: {median_completion_length} characters")

# Limit to 400 entries
if len(dataset_entries) > 400:
    dataset_entries = random.sample(dataset_entries, 400)

# Split into training and validation sets (95% train, 5% validation)
train_size = int(0.95 * len(dataset_entries))
val_size = len(dataset_entries) - train_size
train_entries, val_entries = random_split(dataset_entries, [train_size, val_size])

class ARCCodeDataset(Dataset):
    def __init__(self, entries, tokenizer, functions_context, max_length=256, chunk_size=256, overlap=128):
        self.entries = entries
        self.tokenizer = tokenizer
        self.functions_context = functions_context
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.overlap = overlap

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        prompt = entry['prompt']
        completion = entry['completion']

        # Concatenate functions context, prompt, and completion
        text = self.functions_context + "\n" + prompt + completion

        # Tokenize the entire text without truncation
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=False
        )

        input_ids = encoding['input_ids'].squeeze()

        # Create overlapping chunks
        chunks = []
        start_idx = 0
        while start_idx < len(input_ids):
            end_idx = min(start_idx + self.chunk_size, len(input_ids))
            chunk = input_ids[start_idx:end_idx]
            attention_mask = torch.ones_like(chunk)

            # Create labels (only mask out the prompt part in the first chunk)
            labels = chunk.clone()
            if start_idx == 0:
                prompt_encoding = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                prompt_length = len(prompt_encoding['input_ids'].squeeze())
                labels[:prompt_length] = -100

            # Pad the chunk if it's shorter than chunk_size
            if len(chunk) < self.chunk_size:
                padding_length = self.chunk_size - len(chunk)
                chunk = torch.cat([chunk, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)])

            chunks.append({
                'input_ids': chunk,
                'attention_mask': attention_mask,
                'labels': labels
            })
            start_idx += self.chunk_size - self.overlap

        # Randomly select one chunk for training
        return random.choice(chunks)

class PrintSampleCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset, max_new_tokens=1500, num_beams=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def on_epoch_end(self, args, state, control, **kwargs):
        # Select a random sample from the validation dataset
        sample = random.choice(self.val_dataset)
        input_ids = sample['input_ids'].unsqueeze(0).to(kwargs['model'].device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(kwargs['model'].device)

        # Decode the prompt for display
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"\nDebug: Prompt Length - {len(prompt_text)} characters")
        print(f"Prompt Text (truncated to 500 chars):\n{prompt_text[:500]}...\n")

        # Generate prediction using the model
        with torch.no_grad():
            output_ids = kwargs['model'].generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode the generated output
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion_text = generated_text[len(prompt_text):].strip()

        # Decode the ground truth completion for display
        ground_truth_ids = sample['labels'].masked_fill(sample['labels'] == -100, self.tokenizer.pad_token_id)
        ground_truth_text = self.tokenizer.decode(ground_truth_ids, skip_special_tokens=True)
        ground_truth_completion = ground_truth_text[len(prompt_text):].strip()

        print("\n--- Sample Validation Prediction ---")
        print(f"Prompt (truncated to 500 chars):\n{prompt_text[:500]}...")
        print(f"Expected Completion (truncated to 500 chars):\n{ground_truth_completion[:500]}...")
        print(f"Generated Completion (truncated to 500 chars):\n{completion_text[:500]}...")
        print("-----------------------------------\n")

# Initialize tokenizer and model directly
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Create datasets
train_dataset = ARCCodeDataset(train_entries, tokenizer, functions_context_str)
val_dataset = ARCCodeDataset(val_entries, tokenizer, functions_context_str)

# Define data collator without padding parameter
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Define training arguments with mixed precision and gradient checkpointing
training_args = TrainingArguments(
    gradient_accumulation_steps=8,  # Increased to maintain effective batch size
    fp16=True,  # Enable mixed precision
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,  # Reduced batch size to minimize memory usage per GPU
    per_device_eval_batch_size=1,   # Similarly reduce eval batch size
    eval_steps=100,
    save_steps=500,
    eval_strategy='epoch',
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    ddp_find_unused_parameters=False,
    # Additional settings can be added here if needed
)

# Initialize custom callback
print_callback = PrintSampleCallback(
    tokenizer=tokenizer,
    val_dataset=val_dataset,
    max_new_tokens=1500,  # Adjusted based on maximum completion length
    num_beams=5
)

# Initialize Trainer with all available GPUs
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[print_callback],
)

# Train the model
trainer.train()

# Evaluate on validation set
eval_results = trainer.evaluate()
print(f"Validation Loss: {eval_results['eval_loss']}")

