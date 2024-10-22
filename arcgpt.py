import os
import json
import random
import numpy as np
import warnings
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Setup logging and warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Define constants
DATA_DIR = 'data'
CHALLENGES_FILE = os.path.join(DATA_DIR, 'arc-agi_training_challenges.json')
CODES_FILE = os.path.join(DATA_DIR, 'arc_training_codes.json')
FUNCTIONS_CONTEXT_FILE = os.path.join(DATA_DIR, 'functions_context.json')
SEPARATOR = "<SEP>"

# Utility function to compress a grid
def compress_grid(grid):
    if not grid or not grid[0]:
        return ""
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

# Custom Dataset for training
class ARCCodeDataset(Dataset):
    def __init__(self, entries, tokenizer, chunk_size=512):
        self.entries = entries
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        encoding = self.tokenizer(
            entry['prompt'] + entry['completion'],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.chunk_size
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def load_data():
    with open(CHALLENGES_FILE, 'r') as f:
        challenges = json.load(f)
    with open(CODES_FILE, 'r') as f:
        codes = json.load(f)
    return challenges, codes

def prepare_dataset(challenges, codes, tokenizer):
    entries = []
    for key, code in codes.items():
        if key in challenges:
            challenge = challenges[key]
            examples = challenge.get('train', [])
            if examples:
                prompt_parts = [
                    f"Compressed Input: {compress_grid(ex['input'])}\nCompressed Output: {compress_grid(ex['output'])}"
                    for ex in examples
                ]
                prompt = f"Training Examples:\n{SEPARATOR.join(prompt_parts)}\n\nCode Completion:\n"
                entries.append({'prompt': prompt, 'completion': code})
    return entries

def setup_trainer(model, tokenizer, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=8,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy='epoch',
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        report_to='none'
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    return trainer

def main():
    challenges, codes = load_data()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        device_map='auto',
        torch_dtype=torch.float16
    )

    # Freeze base model parameters and configure LoRA
    for param in model.base_model.parameters():
        param.requires_grad = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    )
    model = get_peft_model(model, lora_config)

    # Prepare datasets
    entries = prepare_dataset(challenges, codes, tokenizer)
    train_size = int(0.9 * len(entries))
    train_entries, val_entries = random_split(entries, [train_size, len(entries) - train_size])

    train_dataset = ARCCodeDataset(train_entries, tokenizer)
    val_dataset = ARCCodeDataset(val_entries, tokenizer)

    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset)
    trainer.train()
    trainer.save_model('./trained_model')

    eval_results = trainer.evaluate()
    print(f"Validation Loss: {eval_results['eval_loss']}")

if __name__ == "__main__":
    main()

