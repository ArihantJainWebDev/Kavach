import numpy as np
import torch
import torch.nn as nn

class TemporalWorldModel(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=128, dropout=0.1, activation="gelu"
        )
        self.temporal = nn.TransformerEncoder(enc, num_layers=layers)
        self.transition = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.state_head = nn.Linear(d_model, n_features)
        self.risk_head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))
        self.stage_head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 5))

    def encode(self, x):
        z = self.temporal(self.input_proj(x))
        return z[:, -1, :]

    def forward(self, x):
        z = self.encode(x)
        z_next = z + self.transition(z)
        next_state = self.state_head(z_next)
        risk = torch.sigmoid(self.risk_head(z_next)).squeeze(-1)
        stage_logits = self.stage_head(z_next)
        return next_state, risk, stage_logits

    @torch.no_grad()
    def rollout(self, x, steps=10):
        self.eval()
        history = x.clone()
        states, risks, stages = [], [], []
        for _ in range(steps):
            next_state, risk, logits = self.forward(history)
            states.append(next_state.cpu().numpy()[0])
            risks.append(float(risk.cpu().item()))
            stages.append(int(logits.argmax(dim=-1).cpu().item()))
            history = torch.cat([history[:, 1:, :], next_state.unsqueeze(1)], dim=1)
        return np.array(states), np.array(risks), np.array(stages)
