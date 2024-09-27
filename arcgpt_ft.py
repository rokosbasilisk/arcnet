import json
import os
import pickle
from typing import List, Tuple, Dict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from termcolor import colored
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Define the constants
MAX_ROW = 30
MAX_COL = 30
NUM_COLORS = 11  # Colors from 0 to 10
MAX_SEQ_LEN = 2048  # Maximum sequence length after chunking
SEP_TOKEN = 9900  # Separator token ID
PAD_TOKEN = 9901  # Padding token ID
VOCAB_SIZE = PAD_TOKEN + 1  # Total vocabulary size

# Define the Action type
Action = Tuple[int, int, int]  # (row, column, value)

# -------------------- Helper Functions --------------------

def action_to_id(action: Action) -> int:
    """
    Encodes an action tuple into a unique integer ID.
    """
    row, col, val = action
    return row * MAX_COL * NUM_COLORS + col * NUM_COLORS + val

def id_to_action(action_id: int) -> Action:
    """
    Decodes an action ID back to an action tuple.
    """
    row = action_id // (MAX_COL * NUM_COLORS)
    remainder = action_id % (MAX_COL * NUM_COLORS)
    col = remainder // NUM_COLORS
    val = remainder % NUM_COLORS
    return (row, col, val)

# -------------------- Data Preparation --------------------

def is_initial_state(grid: List[List[int]]) -> bool:
    """
    Checks if the given grid is the initial state (i.e., all cells are black (0)).
    """
    return all(cell == 0 for row in grid for cell in row)

def get_actions_from_grid(grid: List[List[int]], is_initial: bool = False) -> List[int]:
    """
    Converts a grid into a list of action IDs representing the grid.
    If `is_initial` is True, it skips black (0) cells as they are the default initial state.
    """
    action_ids = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            val = grid[i][j]
            # Skip black cells (0) only if it's the initial state
            if is_initial and val == 0:
                continue
            # Skip invalid positions or values
            if i >= MAX_ROW or j >= MAX_COL or val >= NUM_COLORS:
                continue
            action = (i, j, val)
            action_id = action_to_id(action)
            action_ids.append(action_id)
    return action_ids

def reconstruct_grid_from_actions(action_ids: List[int]) -> List[List[int]]:
    """
    Reconstructs a grid from a list of action IDs.
    Starts with a default black grid (0s) and applies non-black actions.
    """
    # Initialize the grid with all black (0)
    grid = [[0]*MAX_COL for _ in range(MAX_ROW)]
    
    for action_id in action_ids:
        if action_id >= SEP_TOKEN:
            continue  # Skip special tokens
        action = id_to_action(action_id)
        i, j, v = action
        if i < MAX_ROW and j < MAX_COL:
            grid[i][j] = v  # Apply actions (including possibly turning cells back to black)
    
    # Trim the grid to non-zero rows and columns
    max_row = 0
    max_col = 0
    for i in range(MAX_ROW):
        for j in range(MAX_COL):
            if grid[i][j] != 0:
                max_row = max(max_row, i)
                max_col = max(max_col, j)
    trimmed_grid = [row[:max_col+1] for row in grid[:max_row+1]]
    return trimmed_grid


# -------------------- Dataset Class --------------------

class ActionDataset(Dataset):
    def __init__(self, data: List[Dict], max_length: int = MAX_SEQ_LEN):
        self.inputs = [torch.tensor(entry['input'], dtype=torch.long) for entry in data]
        self.targets = [torch.tensor(entry['target'], dtype=torch.long) for entry in data]
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        target_seq = self.targets[idx]

        # Truncate sequences if necessary
        input_seq = input_seq[:self.max_length]
        target_seq = target_seq[:self.max_length]

        return input_seq, target_seq

def collate_fn(batch):
    inputs, targets = zip(*batch)
    # Find the maximum sequence length in the batch
    max_input_length = max([inp.size(0) for inp in inputs])
    max_target_length = max([tgt.size(0) for tgt in targets])
    
    # Pad sequences with PAD_TOKEN to the max length in the batch
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=PAD_TOKEN)

    # Make sure to pad both input and target to the same max length
    max_seq_length = max(max_input_length, max_target_length)

    inputs_padded = nn.functional.pad(inputs_padded, (0, max_seq_length - inputs_padded.size(1)), value=PAD_TOKEN)
    targets_padded = nn.functional.pad(targets_padded, (0, max_seq_length - targets_padded.size(1)), value=PAD_TOKEN)

    return inputs_padded, targets_padded

# -------------------- Model Initialization --------------------

def initialize_pythia_model(vocab_size):
    """
    Initializes the EleutherAI Pythia-70M model and resizes its embeddings to fit the custom vocabulary size.
    """
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped")

    # Resize token embeddings to fit the custom vocabulary size
    model.resize_token_embeddings(vocab_size)
    
    return model

# -------------------- Training and Evaluation Functions --------------------

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]  # Shift the target to input
        tgt_output = tgt[:, 1:]  # Shift the target for output (prediction)

        # Pad tgt_input to the same length as src if necessary
        max_length = src.size(1)
        tgt_input = nn.functional.pad(tgt_input, (0, max_length - tgt_input.size(1)), value=PAD_TOKEN)
        tgt_output = nn.functional.pad(tgt_output, (0, max_length - tgt_output.size(1)), value=PAD_TOKEN)

        # Prepare masks
        src_key_padding_mask = (src == PAD_TOKEN)
        tgt_key_padding_mask = (tgt_input == PAD_TOKEN)

        optimizer.zero_grad()

        # Forward pass
        output = model(input_ids=src, attention_mask=~src_key_padding_mask, labels=tgt_output)

        # The output logits are of shape (batch_size, seq_len, vocab_size)
        logits = output.logits

        # Reshape logits and targets for loss computation
        logits = logits.view(-1, logits.size(-1))  # Shape: (batch_size * seq_len, vocab_size)
        tgt_output = tgt_output.contiguous().view(-1)  # Shape: (batch_size * seq_len)

        # Compute loss, ignoring the PAD_TOKEN in the target
        loss = criterion(logits, tgt_output)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Pad tgt_input and tgt_output to the same length as src if necessary
            max_length = src.size(1)
            tgt_input = nn.functional.pad(tgt_input, (0, max_length - tgt_input.size(1)), value=PAD_TOKEN)
            tgt_output = nn.functional.pad(tgt_output, (0, max_length - tgt_output.size(1)), value=PAD_TOKEN)

            src_key_padding_mask = (src == PAD_TOKEN)
            tgt_key_padding_mask = (tgt_input == PAD_TOKEN)

            output = model(input_ids=src, attention_mask=~src_key_padding_mask, labels=tgt_output)

            logits = output.logits

            # Reshape logits and targets for loss computation
            logits = logits.view(-1, logits.size(-1))  # Shape: (batch_size * seq_len, vocab_size)
            tgt_output = tgt_output.contiguous().view(-1)  # Shape: (batch_size * seq_len)

            loss = criterion(logits, tgt_output)

            total_loss += loss.item()
    return total_loss / len(dataloader)

def plot_losses(train_losses: List[float], val_losses: List[float]):
    plt.figure(figsize=(10,6))
    epochs = range(1, len(train_losses)+1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

# -------------------- Main Function --------------------

def main():
    # Load datasets
    with open('train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('val_dataset.pkl', 'rb') as f:
        val_dataset = pickle.load(f)

    # Create PyTorch datasets
    train_data = ActionDataset(train_dataset)
    val_data = ActionDataset(val_dataset)

    # Create DataLoaders
    batch_size = 16  # Adjust based on GPU memory
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model from Pythia and resize embeddings
    model = initialize_pythia_model(VOCAB_SIZE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # Training loop
    num_epochs = 100  # Adjust as needed
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    # Plot losses
    plot_losses(train_losses, val_losses)

    # Save the model
    model.save_pretrained('fine_tuned_pythia')
    print("Model training completed and saved.")

if __name__ == "__main__":
    main()

