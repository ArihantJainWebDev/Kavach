# KAVACH: Unraveled Forecasting Prototype

KAVACH is a local Streamlit cybersecurity analyst prototype for temporal
network attack-stage forecasting from behavioral telemetry.

The current dashboard workflow is:

1. Replay a historical sequence of behavioral states.
2. Inspect the observed stage and the trained model's next-stage distribution.
3. Explore a recursive forecast trajectory.
4. Review supporting synthetic traffic evidence.
5. Compare stage probabilities and historical telemetry.

This is a research/demo prototype, not a production intrusion-detection or
containment system. The dashboard uses model-derived decision-support scores;
they are not calibrated probabilities of compromise. The evidence explorer is
synthetic UI telemetry and does not represent live network events.

## Current Architecture

```text
Unraveled flow records
				|
				v
17-feature behavioral states
				|
				v
30-step observation windows + chronological train/validation/test splits
				|
				+--> next-stage GRU classifier
				|
				+--> binary transition GRU detector
				|
				v
Streamlit replay dashboard
```

### Dashboard

The main entry point is `app/streamlit_app.py`. It loads:

- `data/processed/states/states.parquet`
- `data/processed/states/labels.parquet`
- `data/processed/splits/feature_scaler.npz`
- `artifacts/gru_hybrid.pt`

The app validates the seven-class order before scoring:

```text
0 Benign
1 Cover up
2 Data Exfiltration
3 Establish Foothold
4 Lateral Movement
5 Reconnaissance
6 Unknown
```

Feature rendering and recursive rollout logic live in
`src/dashboard_features.py`. The current checkpoint has stage-classification
and transition heads, but no feature-regression head. Recursive state updates
therefore use the transition output to interpolate toward observed raw-feature
stage prototypes. This is explicitly documented in the dashboard and should
not be interpreted as a learned next-feature model.

### Feature schema

The trained input vector contains 17 behavioral features, in this order:

```text
flow_count
bytes_sent
bytes_received
unique_destinations
unique_destination_ports
new_peers
syn_ratio
rst_ratio
mean_packet_size
iat_mean
iat_variance
fan_out
active_hosts
total_flows
total_bytes
new_peer_rate
network_fanout
```

## Requirements

- Python 3.10 or newer is recommended.
- The dependencies in `requirements.txt` include Streamlit, Plotly, pandas,
	NumPy, PyTorch, scikit-learn, FastAPI, and Uvicorn.
- A local copy of the processed states, labels, scaler, and GRU checkpoint is
	required to run the current dashboard. Generated binary files are ignored by
	Git; they are not recreated by `pip install`.

## Windows Setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `data/unraveled` is a Git submodule in your clone, initialize it before
running the data-building pipeline:

```powershell
git submodule update --init --recursive
```

If PowerShell blocks activation for the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Run The Dashboard

```powershell
streamlit run app/streamlit_app.py
```

If `streamlit` is not on the global PATH, use the virtual-environment
executable directly:

```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501` by default. To use another
port:

```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py --server.port 8502
```

## Data And Model Pipeline

Run these stages in order when rebuilding the local artifacts. Each command is
run from the repository root.

### 1. Build behavioral states

`src/build_dataset.py` reads flow data from the bundled Unraveled dataset path
`data/unraveled/data/network-flows/` and writes state and label Parquet files
under `data/processed/states/`.

```powershell
python -m src.build_dataset
```

The flow adapter is implemented in `src/state_builder.py`. It aggregates raw
flows into one-minute windows and keeps labels separate from model features.

### 2. Filter and validate states

```powershell
python -m src.filtering
```

This writes filtered state/label files and a filtering report under
`data/processed/filtered/`.

### 3. Build semantic windows

```powershell
python -m src.window_roles
```

The semantic window builder creates 5-, 10-, and 30-step context arrays plus
future stage IDs for rollout evaluation under
`data/processed/windows/semantic/`.

### 4. Create chronological splits and the scaler

```powershell
python -m src.data.create_splits
```

The scaler is fit on training observations only. Split arrays and metadata are
written under `data/processed/splits/`.

### 5. Inspect transition structure or balance training data

```powershell
python -m src.data.transition_baseline
python -m src.data.create_transition_balanced_data
```

The first command evaluates persistence and most-likely-transition baselines.
The second creates a transition-balanced training set for the hybrid model.

### 6. Train the hybrid GRU

```powershell
python src/models/train_gru_hybrid.py
```

The script trains the next-stage classifier and binary transition detector,
selects a transition threshold on validation data, and writes:

- `artifacts/gru_hybrid.pt`
- `artifacts/gru_hybrid_report.json`

The app currently expects the hybrid checkpoint configuration to contain a
17-feature input, seven classes, a 30-step sequence, and compatible GRU
dimensions. Do not change the model architecture without updating both the
training script and dashboard loader together.

## Evaluation Utilities

Useful read-only analysis commands include:

```powershell
python -m src.models.evaluate_persistence
python -m src.data.analyze_transitions
python tools/inspect_labels.py
```

The repository deliberately keeps baseline evaluation visible. In the current
experiments, persistence is a strong baseline and the neural models should not
be described as production-ready or highly accurate without further validation
and calibration work.

## Optional Prototype API

The optional FastAPI service in `src/api.py` uses the older synthetic
`PrototypeForecaster` from `src/pipeline.py`. It is separate from the current
Streamlit dashboard and should not be confused with the trained hybrid GRU
workflow.

```powershell
uvicorn src.api:app --reload
```

Available endpoints are `/health` and `/forecast`. The API is retained as a
prototype surface; the primary user experience is the Streamlit app.

## Repository Layout

```text
app/
	streamlit_app.py             Main analyst dashboard
src/
	state_builder.py             Flow-to-state feature engineering
	filtering.py                 Data quality and relevance filtering
	window_roles.py              Semantic 5/10/30-step windows
	sliding_windows.py           Earlier window-generation utilities
	data/
		create_splits.py           Chronological splits and scaler
		transition_baseline.py     Persistence/transition baselines
		create_transition_balanced_data.py
	models/
		train_gru_hybrid.py        Current hybrid GRU training entry point
		train_gru.py               Standalone next-stage GRU experiment
		train_gru_transition_balanced.py
		train_mlp.py               MLP baseline experiment
	dashboard_features.py        Reusable forecast/evidence UI logic
artifacts/                     Reports and local model checkpoints
data/
	unraveled/                   Unraveled dataset subtree/submodule
	processed/                   Generated states, windows, splits, scalers
tools/                         Small inspection utilities
requirements.txt
```

## Limitations And Safety Notes

- The model predicts the next attack-stage class distribution; it does not
	directly estimate a calibrated compromise probability.
- The transition detector is binary and does not generate a complete next
	feature vector.
- Recursive forecasts compound model and state-update assumptions as the
	horizon grows.
- Synthetic evidence is for UI demonstration only.
- The `Preempt Attack` control displays a containment preview and does not run
	firewall, network, or host-containment actions.
- Raw data, processed arrays, scalers, and checkpoint binaries are local
	artifacts and are intentionally excluded from normal Git commits.

## Development Checks

Run a syntax check before launching the app:

```powershell
python -m py_compile app/streamlit_app.py src/dashboard_features.py
```

Then run the dashboard with the virtual-environment executable if needed:

```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py --server.port 8502
```
