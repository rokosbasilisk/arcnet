import torch
import torch.nn as nn


GRID_SIZE = 30
NUM_COLORS = 10  # 0-9
CONTEXT_LENGTH = 6
BATCH_SIZE = 6
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_LAYERS = 3

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)


# Constants
GRID_SIZE = 30
CELL_SIZE = 20
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
FPS = 30

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_height, max_width):
        super().__init__()
        assert d_model % 2 == 0, "Embedding dimension (d_model) must be even."

        self.d_model = d_model
        self.max_height = max_height
        self.max_width = max_width

        pe_row = torch.zeros(max_height, d_model // 2)
        pe_col = torch.zeros(max_width, d_model // 2)

        position_row = torch.arange(0, max_height, dtype=torch.float).unsqueeze(1)
        position_col = torch.arange(0, max_width, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-torch.log(torch.tensor(10000.0)) / (d_model // 2)))

        pe_row[:, 0::2] = torch.sin(position_row * div_term)
        pe_row[:, 1::2] = torch.cos(position_row * div_term)

        pe_col[:, 0::2] = torch.sin(position_col * div_term)
        pe_col[:, 1::2] = torch.cos(position_col * div_term)

        pe_row = pe_row.unsqueeze(1).expand(-1, max_width, -1)
        pe_col = pe_col.unsqueeze(0).expand(max_height, -1, -1)

        self.register_buffer('pe', torch.cat([pe_row, pe_col], dim=-1))

    def forward(self, x):
        return x + self.pe[:x.size(2), :x.size(3), :]

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
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, max_height, max_width):
        super().__init__()
        self.embedding = nn.Embedding(NUM_COLORS, embed_dim)
        self.pos_encoding = PositionalEncoding2D(embed_dim, max_height, max_width)
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

