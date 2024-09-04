import torch
import torch.nn as nn
import torch.nn.functional as F



NUM_COLORS = 10  # 0-9

class ConvolutionalPositionalEncoding2D(nn.Module):
    def __init__(self, channels, max_height, max_width, num_conv_layers=3):
        super().__init__()
        self.channels = channels
        self.max_height = max_height
        self.max_width = max_width
        
        # Create positional grid
        pos_h = torch.arange(max_height).unsqueeze(1).expand(-1, max_width).unsqueeze(0)
        pos_w = torch.arange(max_width).unsqueeze(0).expand(max_height, -1).unsqueeze(0)
        self.register_buffer('pos_grid', torch.cat([pos_h, pos_w], dim=0).float())
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(2, channels, kernel_size=3, padding=1)
        ])
        for _ in range(num_conv_layers - 1):
            self.conv_layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        batch_size, seq_len, height, width, _ = x.shape
        
        # Generate positional encoding
        pos_encoding = self.pos_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        for conv in self.conv_layers:
            pos_encoding = F.relu(conv(pos_encoding))
        
        # Reshape and add to input
        pos_encoding = pos_encoding.permute(0, 2, 3, 1).unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        x = x + pos_encoding
        return self.norm(x)

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
        self.pos_encoding = ConvolutionalPositionalEncoding2D(embed_dim, max_height, max_width)
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
