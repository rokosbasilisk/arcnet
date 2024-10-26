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

# Disable Weights & Biases (WandB)
os.environ["WANDB_DISABLED"] = "true"

# Optional: Specify CUDA devices if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

# Configure PyTorch's CUDA allocator for better memory handling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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
COT_DATA_FILE = os.path.join(DATA_DIR, 'intermediate_grids_dataset.json')
FUNCTIONS_CONTEXT_FILE = os.path.join(DATA_DIR, 'functions_context.json')
SEPARATOR = "<SEP>"
COMPLETION_TOKEN = "<COMPLETION>"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
batch_size = 8
num_epochs = 40

code_context = """
The compress_grid_optimized function encodes a 2D grid into a string with dimensions and run-length encoding (RLE), while the decompress_grid_optimized function decodes this string back into the original 2D grid format.
"""

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
            labels[:completion_pos[0] + 1] = -100  # Mask up to and including the completion token
        else:
            labels[:] = -100  # If completion token not found, mask all labels
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class PrintCompletionCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset, interval=100):
        """
        Reduced the interval to log every 100 steps to save memory.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        """
        Changed from on_log to on_step_end for better integration.
        """
        if state.global_step % self.interval == 0 and state.global_step > 0:
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
                        max_new_tokens=400,
                        num_beams=3,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.95,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            logger.info(f"\nStep {state.global_step}: Actual vs. Predicted")
            logger.info(f"Input Prompt:\n{input_prompt}\n")
            logger.info(f"Expected Completion:\n{sample['completion']}\n")
            logger.info(f"Generated Completion:\n{generated_text}\n")
            logger.info("--------------------------------------------------\n")

def load_data():
    if not os.path.exists(COT_DATA_FILE):
        logger.error(f"CoT data file not found: {COT_DATA_FILE}")
        return None, None
    if not os.path.exists(FUNCTIONS_CONTEXT_FILE):
        logger.error(f"Functions context file not found: {FUNCTIONS_CONTEXT_FILE}")
        return None, None

    with open(COT_DATA_FILE, 'r') as f:
        cot_data = json.load(f)
    with open(FUNCTIONS_CONTEXT_FILE, 'r') as f:
        functions_context = json.load(f)
    return cot_data, functions_context

def prepare_cot_dataset(cot_data):
    """ Prepare the dataset with chain-of-thought prompts and completions. """
    entries = []
    for item in cot_data:
        input_grid = item['input_grid']
        final_output_grid = item.get('final_output_grid', '')

        # Create prompt parts from intermediate steps
        prompt_parts = [f"Compressed Input: {input_grid}"]
        intermediate_comments = []
        for step in item.get('intermediate_steps', []):
            intermediate_comments.append(f"# Step {step['step']}: {step['grid']}")
        prompt_parts.extend(intermediate_comments)
        prompt_parts.append(f"Compressed Output: {final_output_grid}")

        # Construct the prompt and completion
        prompt = (
            "The model should generate a program that takes the compressed form of an input grid and converts it into the compressed form of the output grid.\n"
            f"the functions used to compress and decompress the grid can be described as follows: {code_context}\n"
            "Below are the transformation steps with intermediate comments:\n"
            f"{SEPARATOR.join(prompt_parts)}\n\nCode Completion:\n"
        )
        completion = item.get('transform_function', 'def transform_grid(I: Grid) -> Grid:\n    return I')
        
        # Replace function name to avoid memorizing hash-based names
        completion = completion.replace(item['hash_id'], "transform_grid")
        
        # Inject intermediate steps into the completion
        for step in item.get('intermediate_steps', []):
            completion = completion.replace(
                step['line'],
                f"{step['line']}\n    # intermediate gridstate: {step['grid']}"
            )
        
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
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        evaluation_strategy='no',
        logging_steps=10,
        learning_rate=5e-5,
        fp16=True,  # Enabled FP16
        save_total_limit=1,
        gradient_accumulation_steps=2,  # Adjusted to balance memory
        gradient_checkpointing=True,     # Enable gradient checkpointing
        deepspeed='./ds_config.json'     # Ensure this path is correct
    )
    pretrainer = Trainer(
        model=model,
        args=pretrain_args,
        train_dataset=pretrain_dataset
    )
    pretrainer.train()
    pretrainer.save_model('./pretrained_model')
    logger.info("Pre-Training Completed.")

def setup_trainer(model, tokenizer, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,  # Batch size per device
        gradient_accumulation_steps=4,          # Accumulate gradients to simulate larger batch
        evaluation_strategy='epoch',
        logging_steps=50,                       # Reduced logging frequency
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,                              # Enabled FP16
        max_grad_norm=1.0,
        warmup_steps=500,
        gradient_checkpointing=True,            # Enable gradient checkpointing
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to='none',
        dataloader_num_workers=4,               # Optimize data loading
        sharded_ddp='simple',                   # Use Sharded DDP for multi-GPU
        bf16=False,                             # Ensure bf16 is disabled if not supported
        deepspeed='./ds_config.json'            # Ensure this path is correct
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
        callbacks=[PrintCompletionCallback(tokenizer=tokenizer, val_dataset=val_dataset, interval=100)]  # Adjusted interval
    )
    return trainer

def main():
    cot_data, functions_context = load_data()
    if cot_data is None or functions_context is None:
        logger.error("Data loading failed. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {'additional_special_tokens': [COMPLETION_TOKEN, SEPARATOR]}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Prepare functions context string
    functions_context_str = "## Function Definitions\n\n" + "\n".join(
        f"**{func['name']}({', '.join(func['arguments'])}) -> {func['return_type']}**: {func['description']}"
        for func in functions_context
    )

    # Load and configure the model without device_map='auto'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Switched to FP16
        # device_map='auto',         # Removed to prevent conflict with DeepSpeed
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

    # Split dataset into training and validation sets
    train_size = int(0.95 * len(entries))
    val_size = len(entries) - train_size
    train_entries, val_entries = random_split(entries, [train_size, val_size])

    # Create datasets
    train_dataset = ARCCodeDataset([entries[i] for i in train_entries.indices], tokenizer)
    val_dataset = ARCCodeDataset([entries[i] for i in val_entries.indices], tokenizer)

    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset)

    # Start training
    try:
        trainer.train()
        trainer.save_model('./trained_model')
        eval_results = trainer.evaluate()
        logger.info(f"Validation Loss: {eval_results['eval_loss']}")
    except torch.cuda.OutOfMemoryError as e:
        logger.error("Training failed due to CUDA Out-Of-Memory.")
        logger.error(str(e))
        # Optionally, implement cleanup or retry mechanisms here
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

