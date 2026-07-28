import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    PyTorch Encoder for Audio Modality.
    Processes 128-dimensional acoustic MFCC feature vectors.
    """

    def __init__(self, input_dim=128, hidden_dim=128, dropout_rate=0.3):
        super(AudioEncoder, self).__init__()
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
            x (torch.Tensor): Audio feature tensor of shape (batch_size, 128)
        Returns:
            torch.Tensor: Latent audio representation of shape (batch_size, hidden_dim)
        """
        return self.net(x)
