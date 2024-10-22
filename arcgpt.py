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

import logging

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = 'data'
CHALLENGES_FILE = os.path.join(DATA_DIR, 'arc-agi_training_challenges.json')
CODES_FILE = os.path.join(DATA_DIR, 'arc_training_codes.json')
FUNCTIONS_CONTEXT_FILE = os.path.join(DATA_DIR, 'functions_context.json')
SEPARATOR = "<SEP>"

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
    def __init__(self, context, tokenizer, max_length=512):
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
    """
    Dataset for ARC code training.
    """
    def __init__(self, entries, tokenizer, chunk_size=512):
        self.entries = entries
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        variable_prompt = entry['prompt']
        completion = entry['completion']
        
        # Tokenize prompt and completion separately for clarity
        prompt_encoding = self.tokenizer(
            variable_prompt,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.chunk_size // 2
        )
        completion_encoding = self.tokenizer(
            completion,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.chunk_size // 2
        )

        input_ids = torch.cat((prompt_encoding['input_ids'], completion_encoding['input_ids']), dim=1).squeeze()
        attention_mask = torch.cat((prompt_encoding['attention_mask'], completion_encoding['attention_mask']), dim=1).squeeze()

        labels = completion_encoding['input_ids'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class PrintSampleCallback(TrainerCallback):
    """
    Callback to generate and print sample outputs after each epoch.
    """
    def __init__(self, tokenizer, val_dataset, max_new_tokens=100, num_beams=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model is None:
            print("Model is not available in kwargs.")
            return

        # Select a random validation sample
        sample = random.choice(self.val_dataset)
        input_ids = sample['input_ids'].unsqueeze(0).to(model.device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(model.device)
        
        # Decode and print the prompt part
        prompt_length = input_ids.shape[1] - sample['labels'].shape[0]
        prompt_ids = input_ids[0, :prompt_length]
        completion_ids = input_ids[0, prompt_length:]
        
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        
        print(f"\n--- Sample Input ---")
        print(f"Prompt Text:\n{prompt_text}")
        print(f"Expected Completion Text:\n{completion_text}")

        # Generate output using the model
        with torch.no_grad():
            output_ids = model.generate(
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
        generated_completion = generated_text[len(prompt_text):].strip()

        print(f"\n--- Generated Output ---")
        print(f"Generated Text:\n{generated_completion}")
        print(f"-----------------------------------\n")

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
    missing_files = []
    for file_path in [CHALLENGES_FILE, CODES_FILE, FUNCTIONS_CONTEXT_FILE]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    if missing_files:
        print(f"Missing data files: {', '.join(missing_files)}")
        return

    with open(CHALLENGES_FILE, 'r') as f:
        challenges = json.load(f)
    with open(CODES_FILE, 'r') as f:
        codes = json.load(f)
    with open(FUNCTIONS_CONTEXT_FILE, 'r') as f:
        functions_context = json.load(f)

    functions_context_str = "## Function Definitions\n\n"
    for func in functions_context:
        args = ", ".join(func["arguments"])
        functions_context_str += f"**{func['name']}({args}) -> {func['return_type']}**: {func['description']}\n\n"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    for param in model.base_model.parameters():
        param.requires_grad = False

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

    pretrain_dataset = PretrainingDataset(functions_context_str, tokenizer, max_length=512)
    pretrain_args = TrainingArguments(
        output_dir='./pretrain_results',
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
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

    function_names = [func['name'] for func in functions_context]
    variable_tokens = [f"x{i}" for i in range(100)]
    custom_tokens = list(set(function_names + variable_tokens))

    tokenizer.add_tokens(custom_tokens)
    model.resize_token_embeddings(len(tokenizer))

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

    train_size = int(0.90 * len(dataset_entries))
    val_size = len(dataset_entries) - train_size
    train_entries, val_entries = random_split(dataset_entries, [train_size, val_size])

    train_dataset = ARCCodeDataset(train_entries, tokenizer, chunk_size=512)
    val_dataset = ARCCodeDataset(val_entries, tokenizer, chunk_size=512)

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=8,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        evaluation_strategy='epoch',
        save_strategy='steps',
        save_steps=500,
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
        max_new_tokens=512,
        num_beams=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CustomDataCollator(tokenizer),
        callbacks=[print_callback],
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Validation Loss: {eval_results['eval_loss']}")

if __name__ == "__main__":
    main()

