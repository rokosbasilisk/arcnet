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

def get_actions_from_grid(grid: List[List[int]]) -> List[int]:
    """
    Converts a grid into a list of action IDs representing the grid.
    """
    action_ids = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            val = grid[i][j]
            if i >= MAX_ROW or j >= MAX_COL or val >= NUM_COLORS:
                continue  # Skip invalid positions or values
            action = (i, j, val)
            action_id = action_to_id(action)
            action_ids.append(action_id)
    return action_ids

def reconstruct_grid_from_actions(action_ids: List[int]) -> List[List[int]]:
    """
    Reconstructs a grid from a list of action IDs.
    """
    grid = [[0]*MAX_COL for _ in range(MAX_ROW)]
    for action_id in action_ids:
        if action_id >= SEP_TOKEN:
            continue  # Skip special tokens
        action = id_to_action(action_id)
        i, j, v = action
        if i < MAX_ROW and j < MAX_COL:
            grid[i][j] = v
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

# -------------------- Test Case and Grid Printing --------------------

def print_grid(grid: List[List[int]]):
    """
    Prints the grid to the console with color representations using termcolor.
    """
    color_mapping = {
        0: 'on_white',
        1: 'on_red',
        2: 'on_green',
        3: 'on_yellow',
        4: 'on_blue',
        5: 'on_magenta',
        6: 'on_cyan',
        7: 'on_grey',
        8: 'on_white',
        9: 'on_red',
        10: 'on_green'
    }
    for row in grid:
        row_str = ''
        for cell in row:
            color = color_mapping.get(cell % 11, 'on_white')
            row_str += colored('  ', 'grey', color)
        print(row_str)
    print("\n")

def test_action_token_conversion(dataset_entry: Dict):
    """
    Tests if the action-to-token conversion and back works correctly.
    Also reconstructs the grids and prints them.
    """
    tokens_input = dataset_entry['input']
    tokens_target = dataset_entry['target']

    # Split input sequence into grids based on SEP_TOKEN
    grids = []
    current_actions = []
    for token in tokens_input:
        if token == SEP_TOKEN:
            if current_actions:
                grid = reconstruct_grid_from_actions(current_actions)
                grids.append(grid)
                current_actions = []
        else:
            current_actions.append(token)
    if current_actions:
        grid = reconstruct_grid_from_actions(current_actions)
        grids.append(grid)

    # Reconstruct target grid
    target_grid = reconstruct_grid_from_actions(tokens_target)

    # Print results
    print("Reconstructed Grids from Input:")
    for idx, grid in enumerate(grids):
        print(f"Grid {idx+1}:")
        print_grid(grid)

    print("Reconstructed Target Grid:")
    print_grid(target_grid)

# -------------------- Data Preparation --------------------

def prepare_dataset(challenges_path: str, solutions_path: str) -> List[Dict]:
    """
    Prepares the dataset by encoding the grids into sequences of action IDs.
    Each dataset entry corresponds to one challenge.
    The input is the concatenation of action IDs representing the training inputs and outputs, separated by SEP_TOKEN, and the test input.
    The target is the action IDs representing the test output.
    """
    dataset = []
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    with open(solutions_path, 'r') as f:
        solutions = json.load(f)

    for key in tqdm(challenges.keys(), desc=f"Processing {os.path.basename(challenges_path)}"):
        challenge = challenges[key]
        solution = solutions.get(key)
        if solution is None:
            print(f"Solution for {key} not found.")
            continue
        train_entries = challenge.get('train', [])
        test_entries = challenge.get('test', [])

        # Build input sequence
        input_sequence = []
        for train_entry in train_entries:
            input_grid = train_entry['input']
            output_grid = train_entry['output']

            # Actions representing the input grid
            tokens_input_grid = get_actions_from_grid(input_grid)
            input_sequence.extend(tokens_input_grid)
            input_sequence.append(SEP_TOKEN)

            # Actions representing the output grid
            tokens_output_grid = get_actions_from_grid(output_grid)
            input_sequence.extend(tokens_output_grid)
            input_sequence.append(SEP_TOKEN)

        # Add test input grid
        for test_entry in test_entries:
            input_grid = test_entry['input']
            tokens_test_input = get_actions_from_grid(input_grid)
            input_sequence.extend(tokens_test_input)
            input_sequence.append(SEP_TOKEN)

        # Build target sequence (test output)
        tokens_test_output = []
        # Some solutions may have multiple outputs (list of grids), take corresponding one
        for idx, test_entry in enumerate(test_entries):
            if isinstance(solution, list) and len(solution) > idx:
                output_grid = solution[idx]
            else:
                output_grid = solution
            tokens_test_output.extend(get_actions_from_grid(output_grid))
            # Assuming only one test output per challenge, so break after first
            break

        # Add to dataset
        dataset.append({'input': input_sequence, 'target': tokens_test_output})

    return dataset

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
    # Pad sequences with PAD_TOKEN
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=PAD_TOKEN)
    return inputs_padded, targets_padded

# -------------------- Transformer Model --------------------

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, padding_idx, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=256, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=MAX_SEQ_LEN)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.generator = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.padding_idx = padding_idx

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        tgt_emb = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        memory = self.transformer.encoder(src_emb.transpose(0,1), src_key_padding_mask=src_key_padding_mask)
        output = self.transformer.decoder(tgt_emb.transpose(0,1), memory, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.generator(output.transpose(0,1))
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=MAX_SEQ_LEN):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:(d_model//2)+1])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            # Expand positional encoding if sequence length exceeds current max_len
            extra_pe = torch.zeros(1, x.size(1) - self.pe.size(1), self.pe.size(2)).to(x.device)
            pe = torch.cat((self.pe, extra_pe), dim=1)
            self.pe = pe
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# -------------------- Training and Evaluation Functions --------------------

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)

        if tgt.size(1) <= 1:
            # Skip this batch due to short target sequence
            print("Skipping batch due to short target sequence.")
            continue

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_key_padding_mask = (src == PAD_TOKEN)
        tgt_key_padding_mask = (tgt_input == PAD_TOKEN)

        # Assertions to check token IDs
        if src.numel() > 0 and tgt_input.numel() > 0:
            assert src.max().item() < VOCAB_SIZE, f"Invalid token ID in src: max {src.max().item()} >= vocab_size {VOCAB_SIZE}"
            assert tgt_input.max().item() < VOCAB_SIZE, f"Invalid token ID in tgt_input: max {tgt_input.max().item()} >= vocab_size {VOCAB_SIZE}"
            assert src.min().item() >= 0, f"Negative token ID in src: min {src.min().item()} < 0"
            assert tgt_input.min().item() >= 0, f"Negative token ID in tgt_input: min {tgt_input.min().item()} < 0"

        optimizer.zero_grad()
        output = model(src, tgt_input, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
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

            if tgt.size(1) <= 1:
                # Skip this batch due to short target sequence
                print("Skipping batch due to short target sequence.")
                continue

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_key_padding_mask = (src == PAD_TOKEN)
            tgt_key_padding_mask = (tgt_input == PAD_TOKEN)

            # Assertions to check token IDs
            if src.numel() > 0 and tgt_input.numel() > 0:
                assert src.max().item() < VOCAB_SIZE, f"Invalid token ID in src: max {src.max().item()} >= vocab_size {VOCAB_SIZE}"
                assert tgt_input.max().item() < VOCAB_SIZE, f"Invalid token ID in tgt_input: max {tgt_input.max().item()} >= vocab_size {VOCAB_SIZE}"
                assert src.min().item() >= 0, f"Negative token ID in src: min {src.min().item()} < 0"
                assert tgt_input.min().item() >= 0, f"Negative token ID in tgt_input: min {tgt_input.min().item()} < 0"

            output = model(src, tgt_input, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
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
    # Prepare datasets
    print("Preparing training dataset...")
    train_dataset = prepare_dataset('data/arc-agi_training_challenges.json', 'data/arc-agi_training_solutions.json')
    print("Preparing validation dataset...")
    val_dataset = prepare_dataset('data/arc-agi_evaluation_challenges.json', 'data/arc-agi_evaluation_solutions.json')

    # Save datasets
    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)

    # Test action-token conversion and print grids
    print("Testing action-token conversion on a sample...")
    sample_entry = random.choice(train_dataset)
    test_action_token_conversion(sample_entry)

    # Create PyTorch datasets
    train_data = ActionDataset(train_dataset)
    val_data = ActionDataset(val_dataset)

    # Create DataLoaders
    batch_size = 32  # Adjust based on GPU memory
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = TransformerModel(vocab_size=VOCAB_SIZE, padding_idx=PAD_TOKEN)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
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

        # Optional: Implement Early Stopping or Model Checkpointing here

    # Plot losses
    plot_losses(train_losses, val_losses)

    # Save the model
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("Model training completed and saved.")

if __name__ == "__main__":
    main()

