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

# Define the Action type
Action = Tuple[int, int, int]  # (row, column, value)

# -------------------- Action Encoder --------------------

class ActionEncoder:
    def __init__(self):
        """
        Initializes the ActionEncoder with empty mappings.
        """
        self.action_to_token: Dict[Action, int] = {}
        self.token_to_action: Dict[int, Action] = {}
        self.next_token_id: int = 1  # Start token IDs from 1
        self.SEP_TOKEN = 0  # Use 0 as the separator token

    def encode_action(self, action: Action) -> int:
        """
        Encodes an action tuple into a unique token ID.
        """
        if action not in self.action_to_token:
            token_id = self.next_token_id
            self.action_to_token[action] = token_id
            self.token_to_action[token_id] = action
            self.next_token_id += 1
        else:
            token_id = self.action_to_token[action]
        return token_id

    def decode_token(self, token_id: int) -> Action:
        """
        Decodes a token ID back to its action tuple.
        """
        if token_id == self.SEP_TOKEN:
            return None  # Separator token
        return self.token_to_action.get(token_id, (0, 0, 0))  # Default to (0,0,0) if not found

    def encode_actions(self, actions: List[Action]) -> List[int]:
        """
        Encodes a list of action tuples into a list of token IDs.
        """
        return [self.encode_action(action) for action in actions]

    def decode_tokens(self, token_ids: List[int]) -> List[Action]:
        """
        Decodes a list of token IDs back to their action tuples.
        """
        return [self.decode_token(token_id) for token_id in token_ids if token_id != self.SEP_TOKEN]

# -------------------- Helper Functions --------------------

def get_actions_from_to(initial_grid: List[List[int]], final_grid: List[List[int]]) -> List[Action]:
    """
    Computes the list of actions required to transform the initial grid to the final grid.
    """
    h = max(len(initial_grid), len(final_grid))
    w = max(len(initial_grid[0]) if initial_grid else 0, len(final_grid[0]) if final_grid else 0)
    # Pad grids to size h x w
    padded_initial = [[0]*w for _ in range(h)]
    padded_final = [[0]*w for _ in range(h)]
    for i in range(len(initial_grid)):
        for j in range(len(initial_grid[0])):
            padded_initial[i][j] = initial_grid[i][j]
    for i in range(len(final_grid)):
        for j in range(len(final_grid[0])):
            padded_final[i][j] = final_grid[i][j]
    # Compute actions
    actions = []
    for i in range(h):
        for j in range(w):
            if padded_initial[i][j] != padded_final[i][j]:
                actions.append((i, j, padded_final[i][j]))
    return actions

def reconstruct_grid_from_actions(actions: List[Action]) -> List[List[int]]:
    """
    Reconstructs a grid from a list of actions.
    """
    h = max(action[0] for action in actions) + 1 if actions else 0
    w = max(action[1] for action in actions) + 1 if actions else 0
    grid = [[0]*w for _ in range(h)]
    for (i, j, v) in actions:
        grid[i][j] = v
    return grid

# -------------------- Data Preparation --------------------

def prepare_dataset(challenges_path: str, solutions_path: str, action_encoder: ActionEncoder) -> List[Dict]:
    """
    Prepares the dataset by encoding the actions into token sequences.
    """
    dataset = []
    challenges = json.load(open(challenges_path, 'r'))
    solutions = json.load(open(solutions_path, 'r'))

    for key in tqdm(challenges.keys(), desc=f"Processing {os.path.basename(challenges_path)}"):
        challenge = challenges[key]
        solution = solutions.get(key)
        if solution is None:
            print(f"Solution for {key} not found.")
            continue
        train_entries = challenge.get('train', [])
        test_entries = challenge.get('test', [])

        # Process train entries
        for train_entry in train_entries:
            input_grid = train_entry['input']
            output_grid = train_entry['output']

            # Actions to go from blank grid to input grid
            actions_input = get_actions_from_to([], input_grid)

            # Actions to go from input grid to output grid
            actions_target = get_actions_from_to(input_grid, output_grid)

            # Encode actions
            tokens_input = action_encoder.encode_actions(actions_input)
            tokens_target = action_encoder.encode_actions(actions_target)

            # Add SEP token
            tokens_input += [action_encoder.SEP_TOKEN]

            # Add to dataset
            dataset.append({'input': tokens_input, 'target': tokens_target})

        # Process test entries
        for idx, test_entry in enumerate(test_entries):
            input_grid = test_entry['input']
            # Some solutions may have multiple outputs (list of grids), take corresponding one
            output_grid = solution[idx] if isinstance(solution, list) and idx < len(solution) else solution

            # Actions to go from blank grid to input grid
            actions_input = get_actions_from_to([], input_grid)

            # Actions to go from input grid to output grid
            actions_target = get_actions_from_to(input_grid, output_grid)

            # Encode actions
            tokens_input = action_encoder.encode_actions(actions_input)
            tokens_target = action_encoder.encode_actions(actions_target)

            # Add SEP token
            tokens_input += [action_encoder.SEP_TOKEN]

            # Add to dataset
            dataset.append({'input': tokens_input, 'target': tokens_target})

    return dataset

# -------------------- Test Case --------------------

def test_action_token_conversion(action_encoder: ActionEncoder, dataset_entry: Dict):
    """
    Tests if the action-to-token conversion and back works correctly.
    """
    tokens_input = dataset_entry['input']
    tokens_target = dataset_entry['target']

    # Decode tokens back to actions
    actions_input = action_encoder.decode_tokens(tokens_input)
    actions_target = action_encoder.decode_tokens(tokens_target)

    # Reconstruct grids from actions
    input_grid_reconstructed = reconstruct_grid_from_actions(actions_input)
    target_grid_reconstructed = reconstruct_grid_from_actions(actions_target)

    # Print results
    print("Decoded Actions Input:", actions_input)
    print("Decoded Actions Target:", actions_target)
    print("Reconstructed Input Grid:")
    for row in input_grid_reconstructed:
        print(row)
    print("Reconstructed Target Grid:")
    for row in target_grid_reconstructed:
        print(row)

    # You can compare with the original grids if you store them in the dataset

# -------------------- Dataset Class --------------------

class ActionDataset(Dataset):
    def __init__(self, data: List[Dict], max_length: int = 512):
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
    # Pad sequences
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)  # Use -100 for ignored index in CrossEntropyLoss
    return inputs_padded, targets_padded

# -------------------- Transformer Model --------------------

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.generator = nn.Linear(d_model, vocab_size)

        self.d_model = d_model

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
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model %2 ==0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:(d_model//2)+1])
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# -------------------- Training and Evaluation Functions --------------------

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt_input == 0)

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
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_key_padding_mask = (src == 0)
            tgt_key_padding_mask = (tgt_input == 0)

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
    # Initialize ActionEncoder
    action_encoder = ActionEncoder()

    # Prepare datasets
    print("Preparing training dataset...")
    train_dataset = prepare_dataset('data/arc-agi_training_challenges.json', 'data/arc-agi_training_solutions.json', action_encoder)
    print("Preparing validation dataset...")
    val_dataset = prepare_dataset('data/arc-agi_evaluation_challenges.json', 'data/arc-agi_evaluation_solutions.json', action_encoder)

    # Save datasets
    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)

    # Save action mappings
    with open('action_mappings.pkl', 'wb') as f:
        pickle.dump({'action_to_token': action_encoder.action_to_token, 'token_to_action': action_encoder.token_to_action}, f)

    # Test action-token conversion
    print("Testing action-token conversion on a sample...")
    sample_entry = random.choice(train_dataset)
    test_action_token_conversion(action_encoder, sample_entry)

    # Create PyTorch datasets
    train_data = ActionDataset(train_dataset)
    val_data = ActionDataset(val_dataset)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    vocab_size = action_encoder.next_token_id  # Total number of tokens
    model = TransformerModel(vocab_size=vocab_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop
    num_epochs = 5
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
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("Model training completed and saved.")

if __name__ == "__main__":
    main()

