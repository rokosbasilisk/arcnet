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

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
EXPRESSIONS_FILE = os.path.join(DATA_DIR, 'expressions.json')
SEPARATOR = "<SEP>"
COMPLETION_TOKEN = "<COMPLETION>"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
batch_size = 10
num_epochs = 100

code_context = """
The compress_grid_optimized function encodes a 2D grid into a string with dimensions and run-length encoding (RLE), while the decompress_grid_optimized function decodes this string back into the original 2D grid format.
"""

class ARCExpressionDataset(Dataset):
    def __init__(self, expressions, tokenizer, chunk_size=512):
        self.expressions = expressions
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        
    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        _, expression = list(self.expressions.items())[idx]
        # Construct the prompt without the hash key, using only the grid data and context
        prompt = (
            "The model should generate a program that takes the compressed form of an input grid and converts it into the compressed form of the output grid.\n"
            f"The functions used to compress and decompress the grid can be described as follows: {code_context}\n"
            "Below is the transformation process with intermediate comments:\n"
            f"{SEPARATOR} Compressed Input:\n\nCode Completion:\n"
        )
        full_text = prompt + COMPLETION_TOKEN + expression
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
            _, expression = list(self.val_dataset.expressions.items())[sample_idx]
            input_prompt = (
                "The model should generate a program that takes the compressed form of an input grid and converts it into the compressed form of the output grid.\n"
                f"The functions used to compress and decompress the grid can be described as follows: {code_context}\n"
                "Below is the transformation process with intermediate comments:\n"
                f"{SEPARATOR} Compressed Input:\n\nCode Completion:\n"
            )
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
            completion = generated_text.split('Code Completion:')[1] if "Code Completion:" in generated_text else generated_text
            logger.info(f"\nStep {state.global_step}: Actual vs. Predicted")
            logger.info(f"Input Prompt:\n{input_prompt}\n")
            logger.info(f"Expected Expression:\n{expression}\n")
            logger.info(f"Generated Expression:\n{generated_text}\n")
            logger.info("--------------------------------------------------\n")

def load_expressions():
    if not os.path.exists(EXPRESSIONS_FILE):
        logger.error(f"Expressions file not found: {EXPRESSIONS_FILE}")
        return None

    with open(EXPRESSIONS_FILE, 'r') as f:
        expressions = json.load(f)
    return expressions

def pretrain_on_context(model, tokenizer, functions_context_str):
    pretrain_prompt = (
        "These functions are from a special DSL called ARC-DSL, used for converting one grid representation to another from the ARC puzzles. Below is the list of function definitions:\n\n"
    )
    pretrain_dataset = ARCExpressionDataset({'context': ''}, tokenizer)
    pretrain_args = TrainingArguments(
        output_dir='./pretrain_results',
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
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
    logger.info("Pre-Training Completed.")

def setup_trainer(model, tokenizer, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        evaluation_strategy='epoch',
        logging_steps=1,
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
        report_to='none'
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
        callbacks=[PrintCompletionCallback(tokenizer=tokenizer, val_dataset=val_dataset, interval=1)]
    )
    return trainer

def main():
    expressions = load_expressions()
    if expressions is None:
        logger.error("Data loading failed. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {'additional_special_tokens': [COMPLETION_TOKEN, SEPARATOR]}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Prepare functions context string
    functions_context_str = "## Function Definitions\n\n" + code_context

    # Load and configure the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use FP32 to avoid FP16 gradient issues
        device_map='auto',
        low_cpu_mem_usage=True,
        use_cache=False
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Pre-train on functions context
    pretrain_on_context(model, tokenizer, functions_context_str)

    # Split dataset into training and validation sets
    expressions_keys = list(expressions.keys())
    train_size = int(0.95 * len(expressions_keys))
    train_keys = random.sample(expressions_keys, train_size)
    val_keys = list(set(expressions_keys) - set(train_keys))
    train_dataset = ARCExpressionDataset({k: expressions[k] for k in train_keys}, tokenizer)
    val_dataset = ARCExpressionDataset({k: expressions[k] for k in val_keys}, tokenizer)

    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset)

    # Start training
    try:
        trainer.train()
        trainer.save_model('./trained_model')
        eval_results = trainer.evaluate()
        logger.info(f"Validation Loss: {eval_results['eval_loss']}")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

