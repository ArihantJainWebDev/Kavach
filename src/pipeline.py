import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from .model import TemporalWorldModel
from .state_builder import STATE_COLUMNS

STAGES = [
    "Reconnaissance",
    "Establish Foothold",
    "Lateral Movement",
    "Data Exfiltration",
    "Cover up",
]

class PrototypeForecaster:
    def __init__(self, states):
        self.scaler = StandardScaler().fit(states)
        self.model = TemporalWorldModel(len(STATE_COLUMNS))
        self._warm_start()

    def _warm_start(self):
        torch.manual_seed(7)
        for p in self.model.parameters():
            if p.ndim > 1:
                torch.nn.init.xavier_uniform_(p, gain=0.5)
            else:
                torch.nn.init.zeros_(p)

    def forecast(self, states, lookback=12, horizon=10):
        scaled = self.scaler.transform(states)
        x = torch.tensor(scaled[-lookback:], dtype=torch.float32).unsqueeze(0)
        future_scaled, _, stage_ids = self.model.rollout(x, horizon)
        future = self.scaler.inverse_transform(future_scaled)

        fan = STATE_COLUMNS.index("network_fanout")
        syn = STATE_COLUMNS.index("syn_ratio")
        trend = np.clip(
            (future[:, fan] - states["network_fanout"].iloc[-1]) /
            (abs(states["network_fanout"].iloc[-1]) + 1e-6), -1, 2
        )
        risk = np.clip(
            0.35 + 0.22*np.arange(1, horizon+1)/horizon
            + 0.18*trend
            + 0.10*(future[:, syn] > states["syn_ratio"].iloc[-1]),
            0, 0.99
        )
        stages = [STAGES[i % len(STAGES)] for i in stage_ids]
        return future, risk, stages

    def evidence(self, states, future, risk, stages):
        latest = states.iloc[-1]
        return [
            f"Forecast horizon: {len(future)} windows.",
            f"Latest network fan-out: {latest['network_fanout']:.3f}.",
            f"Latest SYN ratio: {latest['syn_ratio']:.3f}.",
            f"Projected peak infiltration risk: {risk.max():.0%}.",
            f"Projected terminal stage: {stages[-1]}."
        ]
