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


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "splits"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

INPUT_SIZE = 17
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 7
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    present_labels = np.unique(y_true)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=present_labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def evaluate_model(model, loader):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(sequences)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_predictions)

    metrics = calculate_metrics(y_true, y_pred)

    return metrics, y_true, y_pred


# --------------------------------------------------
# Model
# --------------------------------------------------

class GRUStageClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        output, hidden = self.gru(x)

        # Use the final GRU timestep
        final_state = output[:, -1, :]

        return self.classifier(final_state)


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_split(split_name):
    observations = np.load(
        DATA_DIR / f"observation_windows_{split_name}.npy"
    ).astype(np.float32)

    rollout_stage_ids = np.load(
        DATA_DIR / f"rollout_stage_ids_{split_name}.npy"
    ).astype(np.int64)

    rollout_valid_mask = np.load(
        DATA_DIR / f"rollout_valid_mask_{split_name}.npy"
    ).astype(bool)

    # Forecast the next stage, horizon H1
    targets = rollout_stage_ids[:, 0]
    valid = rollout_valid_mask[:, 0]

    observations = observations[valid]
    targets = targets[valid]

    return observations, targets


# --------------------------------------------------
# Class weights
# --------------------------------------------------

def create_class_weights(targets):
    counts = np.bincount(targets, minlength=NUM_CLASSES)

    weights = np.ones(NUM_CLASSES, dtype=np.float32)

    present = counts > 0

    # Inverse-frequency weighting, but capped to avoid instability
    total = counts[present].sum()
    num_present = present.sum()

    weights[present] = total / (
        num_present * counts[present]
    )

    weights = np.clip(weights, 0.25, 5.0)

    return counts, weights


# --------------------------------------------------
# Training
# --------------------------------------------------

def main():
    set_seed(SEED)

    print(f"Device: {DEVICE}")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print(f"Train observations: {X_train.shape}")
    print(f"Validation observations: {X_val.shape}")
    print(f"Test observations: {X_test.shape}")

    print(f"Train targets: {y_train.shape}")
    print(f"Validation targets: {y_val.shape}")
    print(f"Test targets: {y_test.shape}")

    train_counts, class_weights = create_class_weights(y_train)

    print("\nTraining class counts:")
    for class_id, count in enumerate(train_counts):
        print(f"  Class {class_id}: {count}")

    print("\nClass weights:")
    print(class_weights)

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
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = GRUStageClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    best_val_f1 = -1.0
    best_epoch = -1

    history = []

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        sample_count = 0

        for sequences, targets in train_loader:
            sequences = sequences.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            logits = model(sequences)
            loss = criterion(logits, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = sequences.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        train_loss = running_loss / sample_count

        val_metrics, _, _ = evaluate_model(
            model,
            val_loader,
        )

        scheduler.step(val_metrics["macro_f1"])

        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": current_lr,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_balanced={val_metrics['balanced_accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "num_classes": NUM_CLASSES,
                    "dropout": DROPOUT,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                },
                ARTIFACT_DIR / "gru_next_stage.pt",
            )

    print("\nLoading best model...")
    checkpoint = torch.load(
        ARTIFACT_DIR / "gru_next_stage.pt",
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, y_true, y_pred = evaluate_model(
        model,
        test_loader,
    )

    print("\nBest validation macro F1:")
    print(f"  Epoch: {best_epoch}")
    print(f"  Macro F1: {best_val_f1:.4f}")

    print("\nTest metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(
        f"  Balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=[
                "Benign",
                "Cover up",
                "Data Exfiltration",
                "Establish Foothold",
                "Lateral Movement",
                "Reconnaissance",
                "Unknown",
            ],
            zero_division=0,
        )
    )

    metrics_output = {
        "model": "GRU",
        "task": "next_stage_prediction",
        "device": str(DEVICE),
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_f1),
        "test_metrics": test_metrics,
        "train_class_counts": train_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "history": history,
    }

    with open(
        ARTIFACT_DIR / "gru_next_stage_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics_output, file, indent=2)

    print("\nSaved:")
    print(f"  {ARTIFACT_DIR / 'gru_next_stage.pt'}")
    print(f"  {ARTIFACT_DIR / 'gru_next_stage_metrics.json'}")


if __name__ == "__main__":
    main()