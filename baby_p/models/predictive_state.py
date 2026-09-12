import torch
from torch import nn
from torch.nn import functional as F
from baby_p.env.world import N_ACTIONS


class PredictiveState(nn.Module):
    def __init__(self, latent, hidden, mode):
        super().__init__()
        if mode not in ('recurrent', 'no_history', 'reactive'):
            raise ValueError(f'Invalid state mode: {mode}')
        self.mode, self.hidden = mode, hidden
        inputs = latent + N_ACTIONS
        if mode == 'reactive':
            # Match GRU parameter count within one MLP-width increment.
            gru_parameters = 3 * hidden * (inputs + hidden) + 6 * hidden
            width = max(1, round((gru_parameters - hidden) / (inputs + hidden + 1)))
            self.cell = nn.Sequential(nn.Linear(inputs, width), nn.SiLU(),
                                      nn.Linear(width, hidden), nn.Tanh())
        else:
            self.cell = nn.GRUCell(inputs, hidden)

    def forward(self, z, previous_action, h=None):
        action = F.one_hot(previous_action.long(), N_ACTIONS).to(z.dtype)
        if self.mode == 'reactive':
            return self.cell(torch.cat((z, torch.zeros_like(action)), -1))
        if h is None or self.mode == 'no_history':
            h = z.new_zeros((z.shape[0], self.hidden))
        return self.cell(torch.cat((z, action), -1), h)
