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
from torch.cuda.amp import autocast
import logging

# Set environment variables
os.environ["WANDB_DISABLED"] = "true"

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
COT_FILE = os.path.join(DATA_DIR, 'intermediate_grids_dataset.json')
FUNCTIONS_CONTEXT_FILE = os.path.join(DATA_DIR, 'functions_context.json')
SEPARATOR = "<SEP>"
COMPLETION_TOKEN = "<COMPLETION>"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
batch_size = 8
num_epochs = 8

# Define compress_grid function
def compress_grid(grid):
    """Compress a grid into a run-length encoded string representation."""
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

def decompress_grid(compressed):
    """Decompress a run-length encoded grid string into a 2D list representation."""
    decompressed = []
    current_row = []
    index = 0
    while index < len(compressed):
        char = compressed[index]
        index += 1
        count = ""
        while index < len(compressed) and compressed[index].isdigit():
            count += compressed[index]
            index += 1
        count = int(count)
        current_row.extend([int(char)] * count)
        if len(current_row) == 30:  # Assuming standard 30x30 grid for simplicity
            decompressed.append(current_row)
            current_row = []
    return decompressed

def display_grid(grid):
    """Display a 2D list grid in a readable string format."""
    return "\n".join("".join(str(cell) for cell in row) for row in grid)

# Define the custom dataset for the training
class ARCCodeDataset(Dataset):
    def __init__(self, entries, tokenizer, chunk_size=512):
        self.entries = entries
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
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
        labels = input_ids.clone()
        completion_token_id = self.tokenizer.encode(COMPLETION_TOKEN, add_special_tokens=False)[0]
        completion_pos = (input_ids == completion_token_id).nonzero(as_tuple=True)[0]
        if len(completion_pos) > 0:
            labels[:completion_pos[0]+1] = -100  # Mask up to and including the completion token
        else:
            labels[:] = -100  # If completion token not found, mask all labels
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class PrintCompletionCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset, interval=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.interval = interval

    def on_log(self, args, state, control, **kwargs):
        total_steps = state.global_step
        if total_steps % self.interval == 0 and total_steps > 0:
            model = kwargs.get('model')
            if model is None:
                return

            sample_idx = random.randint(0, len(self.val_dataset) - 1)
            sample = self.val_dataset.entries[sample_idx]
            input_prompt = sample['prompt']
            input_encoding = self.tokenizer(
                input_prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(model.device)
            with torch.no_grad():
                with autocast():
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
            generated_completion = generated_text.split(COMPLETION_TOKEN, 1)[-1].strip()
            
            # Log the actual and generated completions
            logger.info(f"\nStep {state.global_step}: Actual vs. Predicted")
            logger.info(f"Input Prompt:\n{input_prompt}\n")
            logger.info(f"Expected Completion:\n{sample['completion']}\n")
            logger.info(f"Generated Completion:\n{generated_completion}\n")

            # Display the intermediate grids if they exist in the prompt
            for part in input_prompt.split(SEPARATOR):
                if "Step" in part:
                    step_info = part.split(": ", 1)
                    if len(step_info) == 2:
                        step_compressed = step_info[1].strip()
                        decompressed_grid = decompress_grid(step_compressed)
                        logger.info(f"Step {step_info[0]}:\n{display_grid(decompressed_grid)}\n")
            logger.info("--------------------------------------------------\n")

def load_data():
    if not os.path.exists(COT_FILE):
        logger.error(f"COT file not found: {COT_FILE}")
        return None, None
    if not os.path.exists(FUNCTIONS_CONTEXT_FILE):
        logger.error(f"Functions context file not found: {FUNCTIONS_CONTEXT_FILE}")
        return None, None

    with open(COT_FILE, 'r') as f:
        cot_data = json.load(f)
    with open(FUNCTIONS_CONTEXT_FILE, 'r') as f:
        functions_context = json.load(f)
    return cot_data, functions_context

def calculate_length_statistics(entries):
    prompt_lengths = [len(entry['prompt']) for entry in entries]
    completion_lengths = [len(entry['completion']) for entry in entries]
    median_prompt_length = np.median(prompt_lengths)
    max_prompt_length = np.max(prompt_lengths)
    median_completion_length = np.median(completion_lengths)
    max_completion_length = np.max(completion_lengths)

    logger.info(f"Median prompt length: {median_prompt_length}")
    logger.info(f"Max prompt length: {max_prompt_length}")
    logger.info(f"Median completion length: {median_completion_length}")
    logger.info(f"Max completion length: {max_completion_length}")

    # Adjusting max_length based on the computed statistics
    return int(min(max_prompt_length + max_completion_length, 1024))

def prepare_cot_dataset(cot_data):
    entries = []
    for item in cot_data:
        prompt_parts = [f"Compressed Input: {item['input_grid']}"]
        for step in item.get('intermediate_steps', []):
            prompt_parts.append(f"# Step {step['step']}: {step['grid']}")
        prompt_parts.append(f"Compressed Output: {item['final_output_grid']}")
        
        prompt = (
            "The model should generate a program that takes the compressed form of an input grid and converts it into the compressed form of the output grid.\n"
            "Below are the transformation steps with intermediate comments:\n"
            f"{SEPARATOR.join(prompt_parts)}\n\nCode Completion:\n"
        )
        completion = item['transform_function']
        entries.append({'prompt': prompt, 'completion': completion})
    return entries

def pretrain_on_context(model, tokenizer, functions_context_str):
    pretrain_prompt = (
        "These functions are from a special DSL called ARC-DSL, used for converting one grid representation to another from the ARC puzzles. Below is the list of function definitions:\n\n"
    )
    pretrain_dataset = ARCCodeDataset([{'prompt': pretrain_prompt + functions_context_str, 'completion': ''}], tokenizer)
    pretrain_args = TrainingArguments(
        output_dir='./pretrain_results',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        evaluation_strategy='no',
        logging_steps=10,
        learning_rate=5e-5,
        fp16=True,
        save_total_limit=1
    )
    pretrainer = Trainer(
        model=model,
        args=pretrain_args,
        train_dataset=pretrain_dataset
    )
    pretrainer.train()
    pretrainer.save_model('./pretrained_model')
    print("Pre-Training Completed.")

def setup_trainer(model, tokenizer, train_dataset, val_dataset, max_length):
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        evaluation_strategy='epoch',
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        max_grad_norm=1.0,
        warmup_steps=500,
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to='none',
        max_length=max_length
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
        callbacks=[PrintCompletionCallback(tokenizer=tokenizer, val_dataset=val_dataset, interval=10)]
    )
    return trainer

def main():
    # Load CoT dataset
    cot_data, functions_context = load_data()
    if cot_data is None or functions_context is None:
        logger.error("Data loading failed. Exiting.")
        return

    # Prepare functions context string
    functions_context_str = "## Function Definitions\n\n" + "\n".join(
        f"**{func['name']}({', '.join(func['arguments'])}) -> {func['return_type']}**: {func['description']}"
        for func in functions_context
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {'additional_special_tokens': [COMPLETION_TOKEN, SEPARATOR]}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Load and configure the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map='auto',
        low_cpu_mem_usage=True,
        use_cache=False
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Pre-train on functions context
    pretrain_on_context(model, tokenizer, functions_context_str)

    # Prepare dataset
    entries = prepare_cot_dataset(cot_data)
    if len(entries) == 0:
        logger.error("No entries in dataset. Exiting.")
        return

    # Calculate max_length for tokenization based on dataset statistics
    max_length = calculate_length_statistics(entries)

    # Split dataset into training and validation sets
    train_size = int(0.9 * len(entries))
    val_size = len(entries) - train_size
    train_entries, val_entries = random_split(entries, [train_size, val_size])

    # Create datasets
    train_dataset = ARCCodeDataset([entries[i] for i in train_entries.indices], tokenizer, chunk_size=max_length)
    val_dataset = ARCCodeDataset([entries[i] for i in val_entries.indices], tokenizer, chunk_size=max_length)

    # Setup and start training
    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset, max_length)
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

