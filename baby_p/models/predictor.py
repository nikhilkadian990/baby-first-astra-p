import torch
from torch import nn
from torch.nn import functional as F
from baby_p.env.world import N_ACTIONS


class Predictor(nn.Module):
    """Teacher-supplied executed action sequence, never future observations or roles."""
    def __init__(self, hidden, latent):
        super().__init__()
        self.transition = nn.GRUCell(N_ACTIONS, hidden)
        self.readout = nn.Linear(hidden, latent)

    def forward(self, state, actions):
        predictions = {}
        for i in range(actions.shape[1]):
            action = F.one_hot(actions[:, i].long(), N_ACTIONS).to(state.dtype)
            state = self.transition(action, state)
            predictions[i + 1] = self.readout(state)
        return predictions
