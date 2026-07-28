import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    PyTorch Encoder for Text Modality.
    Processes 768-dimensional MentalBERT (mental/mental-bert-base-uncased) text embeddings.
    """

    def __init__(self, input_dim=768, hidden_dim=128, dropout_rate=0.3):
        super(TextEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Text embedding tensor of shape (batch_size, 768)
        Returns:
            torch.Tensor: Latent text representation of shape (batch_size, hidden_dim)
        """
        return self.net(x)
