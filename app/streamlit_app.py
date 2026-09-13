import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATES_PATH = PROJECT_ROOT / "data/processed/states/states.parquet"
LABELS_PATH = PROJECT_ROOT / "data/processed/states/labels.parquet"
MODEL_PATH = PROJECT_ROOT / "data/processed/model/forecast_model.pt"
SCALER_PATH = PROJECT_ROOT / "data/processed/model/scaler.pkl"

DEFAULT_LOOKBACK = 10

STAGE_COLORS = {
    "Benign": "#64748b",
    "Reconnaissance": "#f59e0b",
    "Establish Foothold": "#ef4444",
    "Lateral Movement": "#a855f7",
    "Data Exfiltration": "#dc2626",
    "Cover up": "#334155",
    "Unknown": "#94a3b8",
}


# ============================================================
# MODEL — MUST MATCH src/train_demo.py
# ============================================================

class TemporalForecastModel(nn.Module):

    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        layers=2,
        num_classes=7,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.stage_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):

        x = self.input_projection(x)

        z = self.encoder(x)

        z = z[:, -1]

        logits = self.stage_head(z)

        return logits


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    states = pd.read_parquet(STATES_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    states = states.sort_index()
    labels = labels.loc[states.index]

    return states, labels


@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    classes = checkpoint["classes"]
    lookback = checkpoint["lookback"]
    input_dim = checkpoint["input_dim"]

    model = TemporalForecastModel(
        input_dim=input_dim,
        num_classes=len(classes),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, scaler, classes, lookback


# ============================================================
# FORECAST
# ============================================================

def forecast_at_position(
    states,
    model,
    scaler,
    classes,
    position,
    lookback,
):

    if position < lookback:
        return None

    history = states.iloc[
        position - lookback:position
    ]

    if len(history) != lookback:
        return None

    # Require genuinely consecutive one-minute states.
    times = history.index

    deltas = (
        np.diff(times.values)
        .astype("timedelta64[s]")
        .astype(np.int64)
        / 60.0
    )

    if not np.allclose(deltas, 1.0):
        return None

    x = history.values.astype(np.float32)

    x = scaler.transform(x)

    x = torch.tensor(
        x,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0].numpy()

    predicted_id = int(
        probabilities.argmax()
    )

    predicted_stage = classes[predicted_id]

    probability = float(
        probabilities[predicted_id]
    )

    benign_id = classes.index("Benign")

    risk = float(
        1.0 - probabilities[benign_id]
    )

    return {
        "stage": predicted_stage,
        "probability": probability,
        "risk": risk,
        "probabilities": probabilities,
        "history": history,
    }


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="KAVACH — Network Attack Forecasting",
    page_icon="K",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        max-width: 1500px;
    }

    .hero {
        padding: 18px 22px;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            #111827,
            #1e293b
        );
        color: white;
        margin-bottom: 20px;
    }

    .hero h1 {
        margin-bottom: 4px;
        font-size: 34px;
    }

    .hero p {
        margin: 0;
        color: #cbd5e1;
        font-size: 15px;
    }

    .stage-box {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
    }

    .small-label {
        color: #64748b;
        font-size: 13px;
    }

    .big-value {
        font-size: 25px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD
# ============================================================

try:

    states, labels = load_data()

    model, scaler, classes, lookback = load_model()

except Exception as e:

    st.error(
        "Unable to load the real Unraveled forecasting pipeline."
    )

    st.exception(e)

    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
<h1>KAVACH</h1>
<p>Network attack progression forecasting using temporal behavioural telemetry</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Replay Controls")

st.sidebar.caption(
    "Replay real temporal states generated from the Unraveled dataset."
)

# Prefer the June 30 → July 1 progression for the demo.
preferred_start = pd.Timestamp("2021-06-30 07:20:00")
preferred_end = pd.Timestamp("2021-07-01 13:33:00")

candidate_positions = np.where(
    (states.index >= preferred_start)
    & (states.index <= preferred_end)
)[0]


if len(candidate_positions) > lookback:

    default_position = int(
        candidate_positions[lookback]
    )

    min_position = int(
        candidate_positions[0]
    )

    max_position = int(
        candidate_positions[-1]
    )

else:

    min_position = lookback
    max_position = len(states) - 1
    default_position = min_position


position = st.sidebar.slider(
    "Replay position",
    min_value=min_position,
    max_value=max_position,
    value=min(
        default_position,
        max_position,
    ),
    step=1,
)


current_time = states.index[position]

st.sidebar.markdown("---")

st.sidebar.metric(
    "Replay timestamp",
    current_time.strftime(
        "%Y-%m-%d %H:%M"
    ),
)

st.sidebar.caption(
    "The model uses the previous "
    f"{lookback} consecutive 1-minute "
    "behavioural states to forecast the next state."
)


# ============================================================
# FORECAST
# ============================================================

result = forecast_at_position(
    states=states,
    model=model,
    scaler=scaler,
    classes=classes,
    position=position,
    lookback=lookback,
)


if result is None:

    st.warning(
        "This replay position does not have a complete "
        "10-minute consecutive history."
    )

    st.stop()


history = result["history"]

current_stage = str(
    labels.iloc[position]["stage"]
)

predicted_stage = result["stage"]
prediction_probability = result["probability"]
risk = result["risk"]
probabilities = result["probabilities"]


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Observed Stage",
    current_stage,
)

c2.metric(
    "Forecast: +1 min",
    predicted_stage,
)

c3.metric(
    "Forecast Confidence",
    f"{prediction_probability:.1%}",
)

c4.metric(
    "Non-Benign Probability",
    f"{risk:.1%}",
)


# ============================================================
# FORECAST EXPLANATION
# ============================================================

st.markdown("### Temporal Forecast")

if predicted_stage == "Benign":

    message = (
        "The model currently assigns the highest probability "
        "to a benign next state."
    )

else:

    message = (
        f"The temporal model forecasts "
        f"**{predicted_stage}** as the most likely next stage."
    )

st.info(
    message
    + " Risk is calculated as "
    "**1 − P(Benign)** from the model's stage distribution."
)


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

left, right = st.columns([1.15, 1])


with left:

    st.markdown("### Stage Probability")

    probability_df = pd.DataFrame(
        {
            "Stage": classes,
            "Probability": probabilities,
        }
    ).sort_values(
        "Probability",
        ascending=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=probability_df["Probability"],
            y=probability_df["Stage"],
            orientation="h",
            text=[
                f"{x:.1%}"
                for x in probability_df["Probability"]
            ],
            textposition="auto",
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        xaxis=dict(
            range=[0, 1],
            title="Probability",
        ),
        yaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with right:

    st.markdown("### Attack Progression")

    progression = [
        "Reconnaissance",
        "Establish Foothold",
        "Lateral Movement",
        "Data Exfiltration",
        "Cover up",
    ]

    progression_df = pd.DataFrame(
        {
            "Stage": progression,
            "Status": [
                "Observed / possible"
                if current_stage == stage
                else "Potential future stage"
                for stage in progression
            ],
        }
    )

    st.dataframe(
        progression_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Stage labels are derived from the temporal state "
        "aggregation used to build the demo dataset."
    )


# ============================================================
# BEHAVIOURAL TELEMETRY
# ============================================================

st.markdown("### Behavioural Telemetry")

available_features = [
    "total_bytes",
    "bytes_sent",
    "bytes_received",
    "unique_destinations",
    "new_peers",
    "network_fanout",
    "active_hosts",
    "flow_count",
]

feature_names = {
    "total_bytes": "Total Bytes",
    "bytes_sent": "Bytes Sent",
    "bytes_received": "Bytes Received",
    "unique_destinations": "Unique Destinations",
    "new_peers": "New Peers",
    "network_fanout": "Network Fan-out",
    "active_hosts": "Active Hosts",
    "flow_count": "Flow Count",
}


selected_features = st.multiselect(
    "Signals",
    options=available_features,
    default=[
        "total_bytes",
        "unique_destinations",
        "new_peers",
        "network_fanout",
    ],
    format_func=lambda x: feature_names[x],
)


if selected_features:

    chart = go.Figure()

    for feature in selected_features:

        values = history[feature].values

        chart.add_trace(
            go.Scatter(
                x=history.index,
                y=values,
                mode="lines",
                name=feature_names[feature],
            )
        )

    chart.update_layout(
        height=380,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        xaxis_title="Time",
        yaxis_title="Behavioural value",
        hovermode="x unified",
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )


# ============================================================
# OBSERVED HISTORY
# ============================================================

st.markdown("### Ten-Minute Temporal Context")

context_df = labels.iloc[
    position - lookback:position
][["stage"]].copy()

context_df["time"] = context_df.index

context_df = context_df[
    ["time", "stage"]
]

st.dataframe(
    context_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DATASET / MODEL STATUS
# ============================================================

st.markdown("---")

a, b, c = st.columns(3)

a.metric(
    "Temporal States",
    f"{len(states):,}",
)

b.metric(
    "Behavioural Features",
    f"{states.shape[1]}",
)

c.metric(
    "Model Lookback",
    f"{lookback} min",
)

st.caption(
    "KAVACH demo: real Unraveled-derived temporal states → "
    "10-minute behavioural context → trained temporal Transformer "
    "→ next-minute attack-stage forecast."
)

st.caption(
    "The displayed probabilities are model outputs from the "
    "trained demo checkpoint; they should not be interpreted "
    "as validated production security metrics."
)