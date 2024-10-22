import os
import json
import random
import numpy as np
import warnings
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, 
    DataCollatorForLanguageModeling, TrainerCallback
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
SEPARATOR = "<SEP>"
COMPLETION_TOKEN = "<COMPLETION>"

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

class ARCCodeDataset(Dataset):
    def __init__(self, entries, tokenizer, chunk_size=512):
        self.entries = entries
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        # Combine prompt and completion with a special token
        full_text = entry['prompt'] + COMPLETION_TOKEN + entry['completion']
        
        encoding = self.tokenizer(
            full_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.chunk_size
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create labels: -100 for prompt tokens (they won't contribute to loss)
        labels = input_ids.clone()
        
        # Find the position of the completion token
        completion_token_id = self.tokenizer.encode(COMPLETION_TOKEN, add_special_tokens=False)[0]
        completion_pos = (input_ids == completion_token_id).nonzero(as_tuple=True)[0]
        
        if len(completion_pos) > 0:
            # Mask out the prompt part in labels
            labels[:completion_pos[0]] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class PrintCompletionCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset, interval=10):
        super().__init__()
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0 and state.global_step > 0:
            model = kwargs.get('model')
            if model is None:
                return

            # Pick a random sample from validation dataset
            sample_idx = random.randint(0, len(self.val_dataset) - 1)
            sample = self.val_dataset.entries[sample_idx]
            
            # Get just the prompt part
            input_prompt = sample['prompt']
            
            # Tokenize the prompt
            input_encoding = self.tokenizer(
                input_prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(model.device)

            # Generate completion
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_encoding['input_ids'],
                    attention_mask=input_encoding['attention_mask'],
                    max_new_tokens=200,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Remove the prompt part from generated text
            generated_completion = generated_text[len(input_prompt):]
            
            print(f"\nStep {state.global_step}: Actual vs. Predicted")
            print(f"Input Prompt:\n{input_prompt}\n")
            print(f"Expected Completion:\n{sample['completion']}\n")
            print(f"Generated Completion:\n{generated_completion}\n")
            print("--------------------------------------------------\n")

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
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        evaluation_strategy='epoch',
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        # Modified FP16 training settings
        fp16=True,
        fp16_full_eval=True,
        fp16_backend="auto",
        half_precision_backend="auto",
        bf16=False,  # Disable bfloat16
        # Added gradient clipping
        max_grad_norm=1.0,
        # Added warmup steps
        warmup_steps=500,
        # Added gradient checkpointing
        gradient_checkpointing=True,
        # Other training parameters
        report_to='none',
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[PrintCompletionCallback(tokenizer=tokenizer, val_dataset=val_dataset)]
    )
    return trainer

def main():
    challenges, codes = load_data()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': [COMPLETION_TOKEN]})

    # Modified model loading with proper mixed precision settings
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        device_map='auto',
        torch_dtype=torch.float16,
        # Added low cpu memory usage
        low_cpu_mem_usage=True,
        # Enable gradient checkpointing
        use_cache=False
    )
    
    model.resize_token_embeddings(len(tokenizer))

    # Configure model for training
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    model.enable_input_require_grads()  # Enable input gradients for PEFT
    
    # Freeze base model parameters
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Configure LoRA with adjusted parameters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        bias="none",  # Disable bias training
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()

    # Prepare datasets
    entries = prepare_dataset(challenges, codes, tokenizer)
    train_size = int(0.9 * len(entries))
    train_entries, val_entries = random_split(entries, [train_size, len(entries) - train_size])

    train_dataset = ARCCodeDataset(train_entries, tokenizer)
    val_dataset = ARCCodeDataset(val_entries, tokenizer)

    # Setup and run training
    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset)
    
    try:
        trainer.train()
        trainer.save_model('./trained_model')
        
        eval_results = trainer.evaluate()
        print(f"Validation Loss: {eval_results['eval_loss']}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
