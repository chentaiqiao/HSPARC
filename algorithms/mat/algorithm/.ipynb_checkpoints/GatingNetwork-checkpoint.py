import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F # Added

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GatingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output: [logit_close, logit_open]
        )

    def forward(self, encoded_obs, hard=False):
        logits = self.net(encoded_obs)
        if hard:
            probs = F.softmax(logits, dim=-1)
            indices = torch.argmax(probs, dim=-1, keepdim=True)
            # Create a one-hot tensor but return only the 'open' dimension (index 1)
            # This returns 1.0 if open, 0.0 if closed
            return indices.float(), logits
        else:
            z = F.gumbel_softmax(logits, tau=1.0, hard=True)
            return z[:, :, 1].unsqueeze(-1), logits