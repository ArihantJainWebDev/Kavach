from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATES_PATH = PROJECT_ROOT / "data/processed/states/states.parquet"
LABELS_PATH = PROJECT_ROOT / "data/processed/states/labels.parquet"
MODEL_DIR = PROJECT_ROOT / "data/processed/model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 10
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3

CLASSES = [
    "Benign",
    "Reconnaissance",
    "Establish Foothold",
    "Lateral Movement",
    "Data Exfiltration",
    "Cover up",
    "Unknown",
]

CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

states = pd.read_parquet(STATES_PATH)
labels = pd.read_parquet(LABELS_PATH)

states = states.sort_index()
labels = labels.loc[states.index]

print("States:", states.shape)
print("Labels:", labels.shape)


# ---------------------------------------------------------
# Clean labels
# ---------------------------------------------------------

stage = (
    labels["stage"]
    .astype(str)
    .str.strip()
)

stage_ids = stage.map(CLASS_TO_ID).fillna(CLASS_TO_ID["Unknown"]).astype(int)


# ---------------------------------------------------------
# Chronological split boundaries
# ---------------------------------------------------------

n = len(states)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

print()
print("Split:")
print("Train:", states.index[0], "->", states.index[train_end - 1])
print("Val:  ", states.index[train_end], "->", states.index[val_end - 1])
print("Test: ", states.index[val_end], "->", states.index[-1])


# ---------------------------------------------------------
# Scale using TRAIN only
# ---------------------------------------------------------

scaler = StandardScaler()

X_all = states.values.astype(np.float32)

scaler.fit(X_all[:train_end])

X_scaled = scaler.transform(X_all).astype(np.float32)


# ---------------------------------------------------------
# Build contiguous sequences
# ---------------------------------------------------------

timestamps = states.index

X_train = []
y_train = []

X_val = []
y_val = []

X_test = []
y_test = []


for i in range(LOOKBACK, n):

    # We require the complete history to be consecutive 1-minute states.
    history_times = timestamps[i - LOOKBACK:i + 1]

    deltas = np.diff(history_times.values).astype("timedelta64[s]")
    deltas_minutes = deltas.astype(np.int64) / 60.0

    if not np.allclose(deltas_minutes, 1.0):
        continue

    x = X_scaled[i - LOOKBACK:i]
    y = stage_ids.iloc[i]

    if i < train_end:
        X_train.append(x)
        y_train.append(y)

    elif i < val_end:
        X_val.append(x)
        y_val.append(y)

    else:
        X_test.append(x)
        y_test.append(y)


X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)

X_val = np.asarray(X_val, dtype=np.float32)
y_val = np.asarray(y_val, dtype=np.int64)

X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int64)


print()
print("Sequences:")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)


# ---------------------------------------------------------
# Class weights
# ---------------------------------------------------------

counts = np.bincount(
    y_train,
    minlength=len(CLASSES)
)

print()
print("Training class counts:")

for name, count in zip(CLASSES, counts):
    print(f"{name:25s} {count}")


# Inverse-frequency weights, normalized.
weights = 1.0 / np.maximum(counts, 1)
weights = weights / weights.mean()

class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

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

        # Last timestep = current temporal context
        z = z[:, -1]

        logits = self.stage_head(z)

        return logits


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("Device:", device)


model = TemporalForecastModel(
    input_dim=X_train.shape[-1],
    num_classes=len(CLASSES),
).to(device)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val),
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device),
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)


best_val_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    total = 0
    correct = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(xb)

        preds = logits.argmax(dim=1)

        correct += (preds == yb).sum().item()
        total += len(xb)

    train_loss = total_loss / total
    train_acc = correct / total


    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    val_loss_total = 0
    val_total = 0
    val_correct = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            loss = criterion(logits, yb)

            val_loss_total += loss.item() * len(xb)

            preds = logits.argmax(dim=1)

            val_correct += (
                (preds == yb).sum().item()
            )

            val_total += len(xb)

    val_loss = val_loss_total / val_total
    val_acc = val_correct / val_total


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"train_loss={train_loss:.4f} "
        f"train_acc={train_acc:.3f} "
        f"val_loss={val_loss:.4f} "
        f"val_acc={val_acc:.3f}"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": X_train.shape[-1],
                "classes": CLASSES,
                "lookback": LOOKBACK,
            },
            MODEL_DIR / "forecast_model.pt",
        )

        with open(
            MODEL_DIR / "scaler.pkl",
            "wb",
        ) as f:
            pickle.dump(scaler, f)

        print("  -> saved best model")


print()
print("TRAINING COMPLETE")
print("Model:", MODEL_DIR / "forecast_model.pt")
print("Scaler:", MODEL_DIR / "scaler.pkl")