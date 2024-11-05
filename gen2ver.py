import os
import json
import random
import logging
import argparse
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, TrainerCallback
)
from torch.cuda.amp import autocast
from typing import Dict  # <-- Added import

# Suppress warnings and set up logging
os.environ["WANDB_DISABLED"] = "true"
os.environ["NCCL_P2P_DISABLE"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define constants
SEPARATOR = "<SEP>"
COMPLETION_TOKEN = "<COMPLETION>"
DEFAULT_MODEL_NAME = "distilgpt2"  # Using a smaller model for demonstration
BATCH_SIZE = 8
NUM_EPOCHS = 8
MAX_LENGTH = 512

class FunctionPairDataset(Dataset):
    def __init__(self, data: Dict[str, Dict[str, str]], tokenizer, chunk_size=512):
        """
        Initializes the dataset with function pairs.
        
        Args:
            data (Dict[str, Dict[str, str]]): The function pairs data.
            tokenizer: The tokenizer to use.
            chunk_size (int): Maximum sequence length.
        """
        self.entries = []
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        
        for hash_code, pair in data.items():
            prompt = f"Generator Function Body:\n{pair['generator']}\n\nVerifier Function Body:\n"
            completion = pair['verifier']
            self.entries.append({'prompt': prompt, 'completion': completion})
    
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
            labels[:completion_pos[0] + 1] = -100  # Ignore the prompt in loss
        else:
            labels[:] = -100  # If no completion token found, ignore entire input
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class PrintCompletionCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset, interval=100):
        super().__init__()
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
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
                max_length=MAX_LENGTH
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
            generated_completion = generated_text[len(input_prompt):].strip()
            expected_completion = sample['completion']
            logger.info(f"\nStep {state.global_step}: Actual vs. Predicted")
            logger.info(f"Input Prompt:\n{input_prompt}\n")
            logger.info(f"Expected Completion:\n{expected_completion}\n")
            logger.info(f"Generated Completion:\n{generated_completion}\n")
            logger.info("--------------------------------------------------\n")

def load_json(json_path: str) -> Dict[str, Dict[str, str]]:
    """
    Loads the JSON dataset.
    
    Args:
        json_path (str): Path to the JSON file.
    
    Returns:
        Dict[str, Dict[str, str]]: The loaded data.
    """
    if not os.path.exists(json_path):
        logger.error(f"JSON file not found: {json_path}")
        return {}
    with open(json_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} function pairs from {json_path}")
    return data

def setup_trainer(model, tokenizer, train_dataset, val_dataset):
    """
    Sets up the Trainer with appropriate arguments.
    
    Args:
        model: The pre-trained model.
        tokenizer: The tokenizer.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
    
    Returns:
        Trainer: The configured Trainer instance.
    """
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        evaluation_strategy='epoch',
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True if torch.cuda.is_available() else False,
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
        callbacks=[PrintCompletionCallback(tokenizer=tokenizer, val_dataset=val_dataset, interval=100)]
    )
    return trainer

def main(args):
    set_seed(42)
    
    # Load data
    data = load_json(args.json_path)
    if not data:
        logger.error("No data to train on. Exiting.")
        return
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    special_tokens_dict = {'additional_special_tokens': [COMPLETION_TOKEN, SEPARATOR]}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Prepare dataset
    dataset = FunctionPairDataset(data, tokenizer, chunk_size=MAX_LENGTH)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Training size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset)
    
    # Start training
    try:
        trainer.train()
        trainer.save_model('./trained_model')
        tokenizer.save_pretrained('./trained_model')
        eval_results = trainer.evaluate()
        logger.info(f"Validation Loss: {eval_results['eval_loss']}")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM to predict verifier functions from generator functions")
    parser.add_argument('--json_path', type=str, default='data/function_pairs.json', help='Path to the JSON dataset')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME, help='Pre-trained model name')
    
    args = parser.parse_args()
    
    main(args)

