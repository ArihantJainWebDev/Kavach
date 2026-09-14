import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn

from src.dashboard_features import (
    build_state_prototypes,
    infer_stage_distribution,
    model_historical_risk,
    render_attack_horizon,
    render_evidence_engine,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATES_PATH = PROJECT_ROOT / "data/processed/states/states.parquet"
LABELS_PATH = PROJECT_ROOT / "data/processed/states/labels.parquet"

MODEL_PATH = PROJECT_ROOT / "artifacts/gru_hybrid.pt"
REPORT_PATH = PROJECT_ROOT / "artifacts/gru_hybrid_report.json"
CLASS_MAPPING_REPORT_PATH = (
    PROJECT_ROOT / "artifacts/gru_transition_balanced_report.json"
)
SCALER_PATH = PROJECT_ROOT / "data/processed/splits/feature_scaler.npz"

LOOKBACK = 30

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

EXPECTED_CLASSES = [
    "Benign",
    "Cover up",
    "Data Exfiltration",
    "Establish Foothold",
    "Lateral Movement",
    "Reconnaissance",
    "Unknown",
]


# ============================================================
# MODEL — MUST MATCH src/train_demo.py
# ============================================================

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        output, _ = self.gru(x)
        last_hidden = self.norm(output[:, -1, :])
        return self.classifier(last_hidden)


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


def validate_class_mapping(
    labels: pd.DataFrame,
    classes: list[str],
    config: dict,
) -> None:
    """Fail clearly unless the checkpoint and training label order agree."""
    if classes != EXPECTED_CLASSES:
        raise ValueError(
            "Checkpoint class mapping is not the expected training order: "
            f"{classes!r} != {EXPECTED_CLASSES!r}"
        )
    if int(config["num_classes"]) != len(classes):
        raise ValueError("Checkpoint num_classes does not match its class mapping.")

    observed_labels = set(labels["stage"].astype(str).unique())
    if observed_labels != set(classes):
        raise ValueError(
            "Dataset stage vocabulary does not match the checkpoint mapping: "
            f"{sorted(observed_labels)!r}"
        )

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Training report is required to validate class order: {REPORT_PATH}"
        )
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    report_classes = report.get("class_names")
    if report_classes is None and CLASS_MAPPING_REPORT_PATH.exists():
        mapping_report = json.loads(
            CLASS_MAPPING_REPORT_PATH.read_text(encoding="utf-8")
        )
        report_classes = mapping_report.get("class_names")
    if report_classes != classes:
        raise ValueError(
            "Training report class order does not match the checkpoint mapping: "
            f"{report_classes!r} != {classes!r}"
        )

    benign_index = classes.index("Benign")
    if benign_index != 0:
        raise ValueError(
            f"Unexpected Benign class index {benign_index}; refusing to score risk."
        )
    print("Validated class index-to-label mapping:", dict(enumerate(classes)))
    print("Validated Benign class index:", benign_index)


@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint["config"]
    scaler_data = np.load(SCALER_PATH)

    scaler_mean = scaler_data["mean"].astype(np.float32)
    scaler_scale = scaler_data["scale"].astype(np.float32)

    classes = EXPECTED_CLASSES.copy()

    next_stage_model = GRUClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_size=config["num_classes"],
    )

    transition_model = GRUClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_size=1,
    )

    next_stage_model.load_state_dict(
        checkpoint["next_stage_model_state_dict"]
    )
    transition_model.load_state_dict(
        checkpoint["transition_model_state_dict"]
    )

    next_stage_model.eval()
    transition_model.eval()

    return (
        next_stage_model,
        transition_model,
        scaler_mean,
        scaler_scale,
        classes,
        LOOKBACK,
        float(config.get("threshold", 0.85)),
        config,
    )


# ============================================================
# FORECAST
# ============================================================

def forecast_at_position(
    states,
    next_stage_model,
    transition_model,
    scaler_mean,
    scaler_scale,
    classes,
    position,
    lookback,
    transition_threshold,
):
    if position < lookback:
        return None

    history = states.iloc[position - lookback:position]

    if len(history) != lookback:
        return None

    times = history.index
    deltas = (
        np.diff(times.values)
        .astype("timedelta64[s]")
        .astype(np.int64)
        / 60.0
    )

    if not np.allclose(deltas, 1.0):
        return None

    probabilities = infer_stage_distribution(
        history.to_numpy(dtype=np.float32),
        next_stage_model,
        scaler_mean,
        scaler_scale,
    )
    with torch.no_grad():
        x = torch.tensor(
            (history.to_numpy(dtype=np.float32) - scaler_mean)
            / np.where(scaler_scale == 0, 1.0, scaler_scale),
            dtype=torch.float32,
        ).unsqueeze(0)
        transition_logits = transition_model(x)
        transition_probability = torch.sigmoid(transition_logits)[0].item()

    predicted_id = int(probabilities.argmax())
    predicted_stage = classes[predicted_id]
    probability = float(probabilities[predicted_id])

    benign_id = classes.index("Benign")
    risk = float(np.clip(1.0 - probabilities[benign_id], 0.0, 1.0))

    return {
        "stage": predicted_stage,
        "probability": probability,
        "risk": risk,
        "probabilities": probabilities,
        "transition_probability": transition_probability,
        "transition_detected": transition_probability >= transition_threshold,
        "transition_threshold": transition_threshold,
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
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1560px;
    }

    .hero {
        padding: 20px 24px 18px;
        border: 1px solid #26364b;
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        background: linear-gradient(135deg, #0b1220, #111c2e);
        color: white;
        margin-bottom: 12px;
    }

    .hero h1 {
        margin: 0 0 4px;
        font-size: 32px;
        letter-spacing: 0;
    }

    .hero p {
        margin: 0;
        color: #cbd5e1;
        font-size: 15px;
    }

    .workflow {
        color: #94a3b8;
        font-size: 12px;
        letter-spacing: .02em;
        margin: 0 0 18px 4px;
    }

    [data-testid="stMetric"] {
        background: #111c2e;
        border: 1px solid #26364b;
        border-radius: 8px;
        padding: 10px 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }

    [data-testid="stExpander"] {
        border-color: #26364b;
    }

    .section-rule {
        height: 1px;
        background: #26364b;
        margin: 24px 0 18px;
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

    (
        next_stage_model,
        transition_model,
        scaler_mean,
        scaler_scale,
        classes,
        lookback,
        transition_threshold,
        model_config,
    ) = load_model()
    validate_class_mapping(
        labels=labels,
        classes=classes,
        config=model_config,
    )

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
<div class="workflow">REPLAY TELEMETRY &nbsp;→&nbsp; INSPECT STAGE &nbsp;→&nbsp; FORECAST TRAJECTORY &nbsp;→&nbsp; REVIEW EVIDENCE</div>
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
    "behavioural states to forecast the next stage."
)


# ============================================================
# FORECAST
# ============================================================

result = forecast_at_position(
    states=states,
    next_stage_model=next_stage_model,
    transition_model=transition_model,
    scaler_mean=scaler_mean,
    scaler_scale=scaler_scale,
    classes=classes,
    position=position,
    lookback=lookback,
    transition_threshold=transition_threshold,
)


if result is None:

    st.warning(
        "This replay position does not have a complete "
        f"{lookback}-step consecutive history."
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
historical_times, historical_risk = model_historical_risk(
    states=states,
    position=position,
    lookback=lookback,
    next_stage_model=next_stage_model,
    scaler_mean=scaler_mean,
    scaler_scale=scaler_scale,
    classes=classes,
)
state_prototypes = build_state_prototypes(
    states=states,
    labels=labels,
    classes=classes,
)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Observed Stage",
    current_stage,
)

c2.metric(
    "Forecasted Next Stage",
    predicted_stage,
)

c3.metric(
    "Non-Benign Model Score",
    f"{risk:.1%}",
)

c4.metric(
    "Top-stage Probability",
    f"{prediction_probability:.1%}",
)


# ============================================================
# FORECAST EXPLANATION
# ============================================================

if predicted_stage == "Benign":
    forecast_message = (
        "The model assigns the highest next-stage score to Benign. "
        "The non-benign score remains a decision-support signal, not a calibrated "
        "probability of compromise."
    )
else:
    forecast_message = (
        f"The model's highest next-stage score is **{predicted_stage}**. "
        "The non-benign score is calculated as **1 − P(Benign)** from the stage "
        "distribution and is not a calibrated probability of compromise."
    )

st.caption(forecast_message)

with st.expander("Model diagnostics", expanded=False):
    benign_index = classes.index("Benign")
    st.write("Class mapping", dict(enumerate(classes)))
    st.write("Benign class index", benign_index)
    st.write("Selected probability vector", probabilities.tolist())
    st.write("Probability sum", float(probabilities.sum()))
    st.write(
        "All probabilities valid",
        bool(
            np.all(np.isfinite(probabilities))
            and np.all((probabilities >= 0.0) & (probabilities <= 1.0))
            and np.isclose(probabilities.sum(), 1.0, atol=1e-5)
        ),
    )


# ============================================================
# ATTACK HORIZON
# ============================================================

render_attack_horizon(
    current_stage=current_stage,
    predicted_stage=predicted_stage,
    prediction_probability=prediction_probability,
    current_risk=risk,
    history=history,
    historical_times=historical_times,
    historical_risk=historical_risk,
    next_stage_model=next_stage_model,
    transition_model=transition_model,
    scaler_mean=scaler_mean,
    scaler_scale=scaler_scale,
    classes=classes,
    state_prototypes=state_prototypes,
    transition_threshold=transition_threshold,
    horizon=10,
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

    st.plotly_chart(fig, use_container_width=True)


with right:

    st.markdown("### Attack Progression")

    progression = [
        current_stage,
        predicted_stage,
        "Lateral Movement",
        "Data Exfiltration",
    ]

    progression = list(dict.fromkeys(progression))

    progression_df = pd.DataFrame(
        {
            "Stage": progression,
            "Role": [
                "Observed / current"
                if index == 0
                else "Forecasted next stage"
                if index == 1
                else "Possible downstream stage"
                for index, _ in enumerate(progression)
            ],
        }
    )

    st.dataframe(
        progression_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Downstream stages are deterministic demo context, not multi-step "
        "model predictions."
    )


st.markdown(
    '<div class="section-rule"></div>',
    unsafe_allow_html=True,
)

render_evidence_engine()

st.caption(
    "Synthetic UI telemetry for demonstration only. It will later be "
    "replaced by live network or SIEM events."
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

    st.plotly_chart(chart, use_container_width=True)


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

st.dataframe(context_df, use_container_width=True, hide_index=True)


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
    "30-minute behavioural context → trained hybrid GRU "
    "→ next-minute attack-stage forecast."
)

st.caption(
    "The displayed probabilities are model outputs from the "
    "trained demo checkpoint; they should not be interpreted "
    "as validated production security metrics."
)