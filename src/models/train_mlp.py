from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 7
INPUT_DIM = 17
BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

CLASS_NAMES = [
    "Benign",
    "Cover up",
    "Data Exfiltration",
    "Establish Foothold",
    "Lateral Movement",
    "Reconnaissance",
    "Unknown",
]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_split(split_name):
    features = np.load(
        SPLIT_DIR / f"target_features_{split_name}.npy"
    ).astype(np.float32)

    rollout_ids = np.load(
        SPLIT_DIR / f"rollout_stage_ids_{split_name}.npy"
    ).astype(np.int64)

    rollout_mask = np.load(
        SPLIT_DIR / f"rollout_valid_mask_{split_name}.npy"
    ).astype(bool)

    # Horizon 1 target: next stage after the current observation
    targets = rollout_ids[:, 0]
    valid = rollout_mask[:, 0]

    features = features[valid]
    targets = targets[valid]

    return features, targets


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

print(f"Device: {DEVICE}")
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Val:   {X_val.shape}, {y_val.shape}")
print(f"Test:  {X_test.shape}, {y_test.shape}")


# ---------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------

# Only classes present in the training split receive calculated weights.
# Missing classes cannot be learned without training examples.
class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
present_classes = np.where(class_counts > 0)[0]

class_weights = np.ones(NUM_CLASSES, dtype=np.float32)

for class_id in present_classes:
    class_weights[class_id] = (
        len(y_train)
        / (len(present_classes) * class_counts[class_id])
    )

# Prevent extremely rare classes from dominating the loss.
class_weights = np.clip(class_weights, 0.25, 5.0)

print("\nTraining class counts:")
for class_id, count in enumerate(class_counts):
    print(f"  {class_id}: {CLASS_NAMES[class_id]:22s} {count}")

print("\nClass weights:")
for class_id, weight in enumerate(class_weights):
    print(f"  {class_id}: {CLASS_NAMES[class_id]:22s} {weight:.4f}")


# ---------------------------------------------------------------------
# Datasets and loaders
# ---------------------------------------------------------------------

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train),
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val),
)

test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class NextStageMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


model = NextStageMLP(INPUT_DIM, NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def predict(loader):
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(DEVICE)

            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

            all_targets.append(targets.numpy())
            all_predictions.append(predictions.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)

    return y_true, y_pred


def calculate_metrics(y_true, y_pred):
    # Evaluate macro F1 only over classes actually present in the split.
    present = np.unique(y_true)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=present,
            average="macro",
            zero_division=0,
        ),
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

best_val_f1 = -1.0
best_epoch = -1

print("\nTraining...")
print("-" * 72)

for epoch in range(1, EPOCHS + 1):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for features, targets in train_loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    train_loss = total_loss / total_samples

    train_true, train_pred = predict(train_loader)
    val_true, val_pred = predict(val_loader)

    train_metrics = calculate_metrics(train_true, train_pred)
    val_metrics = calculate_metrics(val_true, val_pred)

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {train_loss:.4f} | "
        f"Train F1 {train_metrics['macro_f1']:.4f} | "
        f"Val F1 {val_metrics['macro_f1']:.4f} | "
        f"Val Bal.Acc {val_metrics['balanced_accuracy']:.4f}"
    )

    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": INPUT_DIM,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1,
            },
            ARTIFACT_DIR / "mlp_next_stage.pt",
        )


# ---------------------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------------------

checkpoint = torch.load(
    ARTIFACT_DIR / "mlp_next_stage.pt",
    map_location=DEVICE,
)

model.load_state_dict(checkpoint["model_state_dict"])

test_true, test_pred = predict(test_loader)
test_metrics = calculate_metrics(test_true, test_pred)

print("\nBest checkpoint")
print("-" * 72)
print(f"Epoch:             {checkpoint['best_epoch']}")
print(f"Validation F1:     {checkpoint['best_val_macro_f1']:.4f}")

print("\nTest metrics")
print("-" * 72)
print(f"Accuracy:          {test_metrics['accuracy']:.4f}")
print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
print(f"Macro F1:          {test_metrics['macro_f1']:.4f}")

print("\nTest classification report")
print("-" * 72)

print(
    classification_report(
        test_true,
        test_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

with open(ARTIFACT_DIR / "mlp_next_stage_metrics.json", "w") as f:
    json.dump(
        {
            "best_epoch": int(checkpoint["best_epoch"]),
            "validation_macro_f1": float(
                checkpoint["best_val_macro_f1"]
            ),
            "test": test_metrics,
        },
        f,
        indent=2,
    )

print(f"\nSaved model to: {ARTIFACT_DIR / 'mlp_next_stage.pt'}")
print(
    f"Saved metrics to: "
    f"{ARTIFACT_DIR / 'mlp_next_stage_metrics.json'}"
)