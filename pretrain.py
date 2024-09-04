import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from tqdm import tqdm

# Constants
GRID_SIZE = 30
NUM_COLORS = 10  # 0-9
CONTEXT_LENGTH = 4
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

class GridDataset(Dataset):
    def __init__(self, trajectories):
        self.data = []
        for trajectory in trajectories:
            states = [state["state"] for state in trajectory["trajectory"]]
            for i in range(len(states) - CONTEXT_LENGTH - 1):
                input_states = states[i:i+CONTEXT_LENGTH]
                target_state = states[i+CONTEXT_LENGTH]
                self.data.append((input_states, target_state))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_states, target_state = self.data[idx]
        input_tensor = torch.tensor(input_states, dtype=torch.long)
        target_tensor = torch.tensor(target_state, dtype=torch.long)
        return input_tensor, target_tensor

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_len=GRID_SIZE):
        super().__init__()
        pe = torch.zeros(CONTEXT_LENGTH, max_len, max_len, d_model)
        for t in range(CONTEXT_LENGTH):
            for pos in range(max_len):
                for i in range(0, d_model, 2):
                    pe[t, pos, :, i] = torch.sin(torch.arange(max_len) / (10000 ** ((2 * i)/d_model)))
                    pe[t, pos, :, i + 1] = torch.cos(torch.arange(max_len) / (10000 ** ((2 * (i + 1))/d_model)))
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :x.size(2), :x.size(3), :]

class SelfAttention2D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x):
        b, t, h, w, c = x.shape
        x = x.view(b, t*h*w, c)
        attn_output, _ = self.mha(x, x, x)
        return attn_output.view(b, t, h, w, c)

class TransformerLayer2D(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = SelfAttention2D(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(x + attended)
        feedforward = self.ff(x.view(-1, x.size(-1))).view(x.shape)
        return self.norm2(x + feedforward)

class GridTransformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.embedding = nn.Embedding(NUM_COLORS, embed_dim)
        self.pos_encoding = PositionalEncoding2D(embed_dim)
        self.layers = nn.ModuleList([TransformerLayer2D(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.final_layer = nn.Linear(embed_dim, NUM_COLORS)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1, :, :, :]  # Only predict the last grid state
        x = self.final_layer(x)
        return x.permute(0, 3, 1, 2)  # Change to (batch, num_colors, height, width)

def load_trajectories(folder_path, num_trajectories=200):
    trajectories = []
    for i in range(1, num_trajectories + 1):
        file_path = os.path.join(folder_path, f"trajectory-{i}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                trajectories.append(json.load(f))
    return trajectories

def exact_match_accuracy(outputs, targets):
    predicted = outputs.argmax(dim=1)
    correct = (predicted == targets).all(dim=(1, 2)).float()
    return correct.mean().item()

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    trajectories = load_trajectories("trajectories")
    train_size = int(0.8 * len(trajectories))
    train_trajectories = trajectories[:train_size]
    val_trajectories = trajectories[train_size:]

    train_dataset = GridDataset(train_trajectories)
    val_dataset = GridDataset(val_trajectories)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = GridTransformer(num_layers=2, embed_dim=64, num_heads=4, ff_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, NUM_COLORS), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += exact_match_accuracy(outputs, targets)

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.reshape(-1, NUM_COLORS), targets.reshape(-1))
                val_loss += loss.item()
                val_acc += exact_match_accuracy(outputs, targets)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), "grid_transformer_model_exact_loss.pth")
    print("Model saved as grid_transformer_model_exact_loss.pth")

if __name__ == "__main__":
    train_model()



