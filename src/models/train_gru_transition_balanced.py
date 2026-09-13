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
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BALANCED_DIR = Path("data/processed/transition_balanced")
SPLIT_DIR = Path("data/processed/splits")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "gru_transition_balanced.pt"
REPORT_PATH = ARTIFACT_DIR / "gru_transition_balanced_report.json"

NUM_CLASSES = 7
INPUT_SIZE = 17
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.30

BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8

CLASS_NAMES = [
    "Benign",
    "Cover up",
    "Data Exfiltration",
    "Establish Foothold",
    "Lateral Movement",
    "Reconnaissance",
    "Unknown",
]


# ============================================================
# Utility functions
# ============================================================

def load_array(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def print_distribution(name, labels):
    values, counts = np.unique(labels, return_counts=True)

    print(f"\n{name} distribution")
    print("-" * 60)

    for value, count in zip(values, counts):
        if 0 <= value < len(CLASS_NAMES):
            label_name = CLASS_NAMES[value]
        else:
            label_name = f"Unknown class {value}"

        print(f"{int(value)} ({label_name}): {int(count)}")


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=list(range(NUM_CLASSES)),
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                labels=list(range(NUM_CLASSES)),
                zero_division=0,
            )
        ),
    }


# ============================================================
# GRU model
# ============================================================

class GRUClassifier(nn.Module):
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

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        output, hidden = self.gru(x)

        # Use the final time-step representation.
        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        logits = self.classifier(last_output)

        return logits


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def predict(model, data_loader):
    model.eval()

    all_predictions = []
    all_targets = []

    for features, targets in data_loader:
        features = features.to(DEVICE)

        logits = model(features)
        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_targets.extend(targets.numpy().tolist())

    return np.array(all_targets), np.array(all_predictions)


def evaluate_model(model, data_loader, split_name):
    y_true, y_pred = predict(model, data_loader)

    metrics = calculate_metrics(y_true, y_pred)

    print(f"\n{split_name} metrics")
    print("-" * 60)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:       {metrics['weighted_f1']:.4f}")

    print(f"\n{split_name} classification report")
    print("-" * 60)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print(report_text)

    print(f"{split_name} confusion matrix")
    print("-" * 60)
    print(confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    ))

    return y_true, y_pred, metrics


# ============================================================
# Main training procedure
# ============================================================

def main():
    print("=" * 70)
    print("Training GRU on transition-balanced data")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Balanced data directory: {BALANCED_DIR}")
    print(f"Model output: {MODEL_PATH}")

    # --------------------------------------------------------
    # Load balanced training data
    # --------------------------------------------------------

    X_train = load_array(
        BALANCED_DIR / "observation_windows_train.npy"
    )

    y_train = load_array(
        BALANCED_DIR / "next_stage_train.npy"
    )

    # --------------------------------------------------------
    # Load original validation and test data
    # --------------------------------------------------------

    X_val = load_array(
        SPLIT_DIR / "observation_windows_val.npy"
    )

    rollout_val = load_array(
        SPLIT_DIR / "rollout_stage_ids_val.npy"
    )

    X_test = load_array(
        SPLIT_DIR / "observation_windows_test.npy"
    )

    rollout_test = load_array(
        SPLIT_DIR / "rollout_stage_ids_test.npy"
    )

    # The first rollout step is the H1 target.
    y_val = rollout_val[:, 0]
    y_test = rollout_test[:, 0]

    # Remove invalid target rows if any.
    train_mask = y_train >= 0
    val_mask = y_val >= 0
    test_mask = y_test >= 0

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    X_val = X_val[val_mask]
    y_val = y_val[val_mask]

    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    print("\nDataset shapes")
    print("-" * 60)
    print(f"Training features:   {X_train.shape}")
    print(f"Training targets:    {y_train.shape}")
    print(f"Validation features: {X_val.shape}")
    print(f"Validation targets:  {y_val.shape}")
    print(f"Test features:       {X_test.shape}")
    print(f"Test targets:        {y_test.shape}")

    print_distribution("Training", y_train)
    print_distribution("Validation", y_val)
    print_distribution("Test", y_test)

    # --------------------------------------------------------
    # Convert to PyTorch tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long,
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.long,
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long,
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor,
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor,
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

    # --------------------------------------------------------
    # Calculate class weights from balanced training targets
    # --------------------------------------------------------

    class_counts = np.bincount(
        y_train,
        minlength=NUM_CLASSES,
    ).astype(np.float32)

    class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)

    present_classes = class_counts > 0

    if present_classes.any():
        total_present = class_counts[present_classes].sum()

        class_weights[present_classes] = (
            total_present
            / (
                present_classes.sum()
                * class_counts[present_classes]
            )
        )

    # Avoid zero weights for absent classes.
    class_weights[class_weights == 0] = 1.0

    # Cap very large weights.
    class_weights = np.clip(
        class_weights,
        0.25,
        5.0,
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print("\nClass weights")
    print("-" * 60)

    for index, weight in enumerate(class_weights):
        print(
            f"{index} ({CLASS_NAMES[index]}): "
            f"{weight:.4f}"
        )

    # --------------------------------------------------------
    # Create model, loss, optimizer
    # --------------------------------------------------------

    model = GRUClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor
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
        patience=3,
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_val_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        sample_count = 0

        for features, targets in train_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = features.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        train_loss = running_loss / max(sample_count, 1)

        y_val_true, y_val_pred = predict(
            model,
            val_loader,
        )

        val_metrics = calculate_metrics(
            y_val_true,
            y_val_pred,
        )

        scheduler.step(val_metrics["macro_f1"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Balanced: {val_metrics['balanced_accuracy']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": float(current_lr),
        })

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "num_classes": NUM_CLASSES,
                    "dropout": DROPOUT,
                    "class_names": CLASS_NAMES,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_epoch": best_epoch,
                },
                MODEL_PATH,
            )

            print(
                f"  Saved best model at epoch {epoch} "
                f"with validation Macro F1 "
                f"{best_val_macro_f1:.4f}"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}"
            )
            break

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print("\nLoading best model...")
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch: {checkpoint['best_epoch']}"
    )

    print(
        f"Best validation Macro F1: "
        f"{checkpoint['best_val_macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    y_val_true, y_val_pred, val_metrics = evaluate_model(
        model,
        val_loader,
        "Validation",
    )

    y_test_true, y_test_pred, test_metrics = evaluate_model(
        model,
        test_loader,
        "Test",
    )

    # --------------------------------------------------------
    # Evaluate only actual transitions in the test set
    # --------------------------------------------------------

    current_test = load_array(
        SPLIT_DIR / "target_stage_ids_test.npy"
    )

    current_test = current_test[test_mask]

    transition_mask = (
        current_test != y_test_true
    )

    transition_count = int(transition_mask.sum())

    print("\nActual-transition evaluation")
    print("-" * 60)
    print(f"Actual transitions: {transition_count}")
    print(
        f"Transition ratio: "
        f"{transition_count / len(y_test_true):.4f}"
    )

    transition_metrics = None

    if transition_count > 0:
        transition_true = y_test_true[transition_mask]
        transition_pred = y_test_pred[transition_mask]

        transition_metrics = calculate_metrics(
            transition_true,
            transition_pred,
        )

        print(
            f"Transition accuracy: "
            f"{transition_metrics['accuracy']:.4f}"
        )

        print(
            f"Transition balanced accuracy: "
            f"{transition_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Transition Macro F1: "
            f"{transition_metrics['macro_f1']:.4f}"
        )

        print("\nTransition classification report")
        print("-" * 60)

        print(
            classification_report(
                transition_true,
                transition_pred,
                labels=list(range(NUM_CLASSES)),
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {
        "device": str(DEVICE),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_macro_f1": float(
            checkpoint["best_val_macro_f1"]
        ),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "transition_metrics": transition_metrics,
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "class_names": CLASS_NAMES,
        "history": history,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\nArtifacts saved")
    print("-" * 60)
    print(f"Model:  {MODEL_PATH}")
    print(f"Report: {REPORT_PATH}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()