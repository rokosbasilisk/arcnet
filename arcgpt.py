import os
import json
import random
import numpy as np
import warnings
from pathlib import Path

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

warnings.filterwarnings("ignore", category=FutureWarning)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = 'data'
CHALLENGES_FILE = os.path.join(DATA_DIR, 'arc-agi_training_challenges.json')
CODES_FILE = os.path.join(DATA_DIR, 'arc_training_codes.json')
FUNCTIONS_CONTEXT_FILE = os.path.join(DATA_DIR, 'functions_context.json')
SEPARATOR = "\n===\n"

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

class PretrainingDataset(Dataset):
    def __init__(self, context, tokenizer, max_length=1024):
        self.context = context
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.context,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class ARCCodeDataset(Dataset):
    def __init__(self, entries, tokenizer, chunk_size=1024):
        self.entries = entries
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        variable_prompt = entry['prompt']
        completion = entry['completion']
        encoding = self.tokenizer(
            variable_prompt + completion,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.chunk_size,
            return_attention_mask=True
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        prompt_encoding = self.tokenizer(
            variable_prompt,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.chunk_size
        )
        prompt_length = (prompt_encoding['input_ids'] != self.tokenizer.pad_token_id).sum().item()
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class DebugLossCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        loss = kwargs.get('loss')
        if loss is not None:
            print(f"Debug: Loss={loss.item()}, requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}")

class PrintSampleCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset, max_new_tokens=1500, num_beams=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def on_epoch_end(self, args, state, control, **kwargs):
        sample = random.choice(self.val_dataset)
        input_ids = sample['input_ids'].unsqueeze(0).to(kwargs['model'].device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(kwargs['model'].device)
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"\nDebug: Prompt Length - {len(prompt_text)} characters")
        print(f"Prompt Text (truncated to 500 chars):\n{prompt_text[:500]}...\n")
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
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion_text = generated_text[len(prompt_text):].strip()
        ground_truth_ids = sample['labels'].masked_fill(sample['labels'] == -100, self.tokenizer.pad_token_id)
        ground_truth_text = self.tokenizer.decode(ground_truth_ids, skip_special_tokens=True)
        ground_truth_completion = ground_truth_text[len(prompt_text):].strip()
        print("\n--- Sample Validation Prediction ---")
        print(f"Prompt (truncated to 500 chars):\n{prompt_text[:500]}...")
        print(f"Expected Completion (truncated to 500 chars):\n{ground_truth_completion[:500]}...")
        print(f"Generated Completion (truncated to 500 chars):\n{completion_text[:500]}...")
        print("-----------------------------------\n")

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def main():
    if not os.path.exists(CHALLENGES_FILE):
        print(f"Challenges file not found: {CHALLENGES_FILE}")
        return
    if not os.path.exists(CODES_FILE):
        print(f"Codes file not found: {CODES_FILE}")
        return
    if not os.path.exists(FUNCTIONS_CONTEXT_FILE):
        print(f"Functions context file not found: {FUNCTIONS_CONTEXT_FILE}")
        return

    print("Loading challenges...")
    with open(CHALLENGES_FILE, 'r') as f:
        challenges = json.load(f)

    print("Loading codes...")
    with open(CODES_FILE, 'r') as f:
        codes = json.load(f)

    print("Loading functions context...")
    with open(FUNCTIONS_CONTEXT_FILE, 'r') as f:
        functions_context = json.load(f)

    functions_context_str = "## Function Definitions\n\n"
    for func in functions_context:
        args = ", ".join(func["arguments"])
        functions_context_str += f"**{func['name']}({args}) -> {func['return_type']}**: {func['description']}\n\n"

    print("Loading tokenizer and model with device_map='auto'...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print("Freezing base model parameters...")
    for param in model.base_model.parameters():
        param.requires_grad = False

    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj"
        ]
    )
    model = get_peft_model(model, lora_config)

    # Ensure LoRA parameters require gradients
    print("Ensuring LoRA parameters require gradients...")
    trainable_params = []
    non_trainable_params = []
    for name, param in model.named_parameters():
        if 'lora_up' in name or 'lora_down' in name:
            param.requires_grad = True
            trainable_params.append(name)
        else:
            non_trainable_params.append(name)

    print("\n--- Trainable Parameters After LoRA Application and Freezing Base Model ---")
    print(f"Total Trainable Parameters: {len(trainable_params)}")
    print(f"Total Non-Trainable Parameters: {len(non_trainable_params)}\n")

    print("Sample Trainable Parameters:")
    for name in trainable_params[:10]:
        print(f" - {name}")

    if not trainable_params:
        print("No LoRA parameters are trainable. Exiting.")
        return

    print("Pre-Training on constant context...")
    pretrain_dataset = PretrainingDataset(functions_context_str, tokenizer, max_length=1024)
    pretrain_args = TrainingArguments(
        output_dir='./pretrain_results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        evaluation_strategy='no',
        save_strategy='epoch',
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=True,
        report_to='none',
    )
    pretrain_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    pretrainer = Trainer(
        model=model,
        args=pretrain_args,
        train_dataset=pretrain_dataset,
        data_collator=pretrain_data_collator,
    )
    pretrainer.train()
    pretrainer.save_model('./pretrained_model')
    print("Pre-Training Completed.\n")

    print("Preparing main training dataset...")
    dataset_entries = []

    for key, code in codes.items():
        if key not in challenges:
            continue
        challenge = challenges[key]
        train_examples = challenge.get('train', [])
        if not train_examples:
            continue
        prompt_parts = []
        for example in train_examples:
            input_grid = example.get('input', [])
            output_grid = example.get('output', [])
            compressed_input = compress_grid(input_grid)
            compressed_output = compress_grid(output_grid)
            prompt_parts.append(f"Compressed Input: {compressed_input}\nCompressed Output: {compressed_output}")
        variable_prompt = f"Training Examples:\n" + f"{SEPARATOR}".join(prompt_parts) + "\n\nCode Completion:\n"
        dataset_entries.append({
            'prompt': variable_prompt,
            'completion': code
        })

    prompt_lengths = [len(entry['prompt']) for entry in dataset_entries]
    completion_lengths = [len(entry['completion']) for entry in dataset_entries]

    max_prompt_length = max(prompt_lengths) if prompt_lengths else 0
    median_prompt_length = np.median(prompt_lengths) if prompt_lengths else 0
    max_completion_length = max(completion_lengths) if completion_lengths else 0
    median_completion_length = np.median(completion_lengths) if completion_lengths else 0

    print(f"Maximum Prompt Length: {max_prompt_length} characters")
    print(f"Median Prompt Length: {median_prompt_length} characters")
    print(f"Maximum Completion Length: {max_completion_length} characters")
    print(f"Median Completion Length: {median_completion_length} characters")

    if len(dataset_entries) > 400:
        dataset_entries = random.sample(dataset_entries, 400)
        print(f"Limited dataset to 400 entries. New dataset size: {len(dataset_entries)}")
    else:
        print(f"Dataset size: {len(dataset_entries)}")

    train_size = int(0.95 * len(dataset_entries))
    val_size = len(dataset_entries) - train_size
    train_entries, val_entries = random_split(dataset_entries, [train_size, val_size])

    print(f"Training set size: {len(train_entries)}")
    print(f"Validation set size: {len(val_entries)}")

    train_dataset = ARCCodeDataset(train_entries, tokenizer, chunk_size=1024)
    val_dataset = ARCCodeDataset(val_entries, tokenizer, chunk_size=1024)

    print("\n--- Sample from Training Dataset ---")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"Sample {i+1}:")
        print("Input IDs:", sample['input_ids'])
        print("Attention Mask:", sample['attention_mask'])
        print("Labels:", sample['labels'])
        print()
    
    print("\n--- Sample from Validation Dataset ---")
    for i in range(min(3, len(val_dataset))):
        sample = val_dataset[i]
        print(f"Sample {i+1}:")
        print("Input IDs:", sample['input_ids'])
        print("Attention Mask:", sample['attention_mask'])
        print("Labels:", sample['labels'])
        print()

    print("\n--- Label Verification ---")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        num_valid_labels = (sample['labels'] != -100).sum().item()
        print(f"Train Sample {i+1}: Number of valid labels: {num_valid_labels}")
    for i in range(min(3, len(val_dataset))):
        sample = val_dataset[i]
        num_valid_labels = (sample['labels'] != -100).sum().item()
        print(f"Validation Sample {i+1}: Number of valid labels: {num_valid_labels}")
    print()

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        evaluation_strategy='epoch',
        save_strategy='steps',
        save_steps=500,
        eval_steps=100,
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,
        report_to='none',
        run_name='arcnet_training',
        ddp_find_unused_parameters=False,
    )

    print_callback = PrintSampleCallback(
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        max_new_tokens=1500,
        num_beams=5
    )

    debug_callback = DebugLossCallback()

    print("Initializing Trainer for main training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CustomDataCollator(tokenizer),
        callbacks=[print_callback, debug_callback],
    )

    if len(train_dataset) == 0:
        print("Training dataset is empty. Exiting.")
        return

    if len(val_dataset) == 0:
        print("Validation dataset is empty. Exiting.")
        return

    print("\n=== Starting Main Training Phase ===")
    trainer.train()
    print("=== Main Training Completed ===\n")

    eval_results = trainer.evaluate()
    print(f"Validation Loss: {eval_results['eval_loss']}")

if __name__ == "__main__":
    main()

