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


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "splits"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

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

# Candidate thresholds for transition detection.
# Higher threshold = fewer predicted transitions.
THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Model
# ============================================================

class GRUNextStage(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
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
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        output, _ = self.gru(x)

        # Use the final observation in the 30-step window.
        final_state = output[:, -1, :]
        final_state = self.norm(final_state)

        logits = self.classifier(final_state)
        return logits


class TransitionDetector(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
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
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        output, _ = self.gru(x)

        final_state = output[:, -1, :]
        final_state = self.norm(final_state)

        logits = self.classifier(final_state).squeeze(-1)
        return logits


# ============================================================
# Data loading
# ============================================================

def load_split(split: str):
    observations = np.load(
        DATA_DIR / f"observation_windows_{split}.npy"
    ).astype(np.float32)

    current_stage = np.load(
        DATA_DIR / f"target_stage_ids_{split}.npy"
    ).astype(np.int64)

    rollout_ids = np.load(
        DATA_DIR / f"rollout_stage_ids_{split}.npy"
    ).astype(np.int64)

    # First rollout stage is the actual next-stage target.
    next_stage = rollout_ids[:, 0]

    valid = (
        (current_stage >= 0)
        & (next_stage >= 0)
        & (current_stage < NUM_CLASSES)
        & (next_stage < NUM_CLASSES)
    )

    observations = observations[valid]
    current_stage = current_stage[valid]
    next_stage = next_stage[valid]

    transition_target = (current_stage != next_stage).astype(np.float32)

    return (
        observations,
        current_stage,
        next_stage,
        transition_target,
    )


# ============================================================
# Batch utilities
# ============================================================

def iterate_batches(x, y, batch_size, shuffle=False):
    indices = np.arange(len(x))

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]

        xb = torch.from_numpy(x[batch_indices])
        yb = torch.from_numpy(y[batch_indices])

        yield xb, yb


# ============================================================
# Evaluation helpers
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def evaluate_stage_predictions(
    y_true,
    y_pred,
    split_name,
):
    metrics = calculate_metrics(y_true, y_pred)

    print(f"\n{split_name} stage metrics:")
    print(json.dumps(metrics, indent=2))

    print(f"\n{split_name} classification report:")
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

    print(f"{split_name} confusion matrix:")
    print(confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    ))

    return metrics


def evaluate_transition_predictions(
    current_stage,
    next_stage,
    predicted_transition,
    split_name,
):
    actual_transition = current_stage != next_stage

    transition_metrics = {
        "transition_accuracy": float(
            accuracy_score(actual_transition, predicted_transition)
        ),
        "transition_balanced_accuracy": float(
            balanced_accuracy_score(
                actual_transition,
                predicted_transition,
            )
        ),
        "transition_f1": float(
            f1_score(
                actual_transition,
                predicted_transition,
                zero_division=0,
            )
        ),
    }

    print(f"\n{split_name} transition detector metrics:")
    print(json.dumps(transition_metrics, indent=2))

    print("\nTransition confusion matrix:")
    print(confusion_matrix(
        actual_transition,
        predicted_transition,
        labels=[0, 1],
    ))

    return transition_metrics


# ============================================================
# Model inference
# ============================================================

@torch.no_grad()
def predict_next_stage_logits(model, x, device):
    model.eval()

    outputs = []

    for start in range(0, len(x), BATCH_SIZE):
        xb = torch.from_numpy(
            x[start:start + BATCH_SIZE]
        ).to(device)

        logits = model(xb)
        outputs.append(logits.cpu().numpy())

    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def predict_transition_probabilities(model, x, device):
    model.eval()

    outputs = []

    for start in range(0, len(x), BATCH_SIZE):
        xb = torch.from_numpy(
            x[start:start + BATCH_SIZE]
        ).to(device)

        logits = model(xb)
        probabilities = torch.sigmoid(logits)

        outputs.append(probabilities.cpu().numpy())

    return np.concatenate(outputs, axis=0)


# ============================================================
# Training: next-stage classifier
# ============================================================

def train_next_stage_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    device,
):
    class_counts = np.bincount(
        y_train,
        minlength=NUM_CLASSES,
    ).astype(np.float32)

    class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)

    nonzero = class_counts > 0
    class_weights[nonzero] = (
        len(y_train)
        / (NUM_CLASSES * class_counts[nonzero])
    )

    # Keep absent classes neutral.
    class_weights[~nonzero] = 1.0

    # Avoid extreme weights.
    class_weights = np.clip(
        class_weights,
        0.25,
        5.0,
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    print("\nNext-stage class counts:")
    print(class_counts.astype(int))

    print("Next-stage class weights:")
    print(class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total_items = 0

        for xb, yb in iterate_batches(
            x_train,
            y_train,
            BATCH_SIZE,
            shuffle=True,
        ):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            running_loss += loss.item() * len(yb)
            total_items += len(yb)

        val_logits = predict_next_stage_logits(
            model,
            x_val,
            device,
        )

        val_pred = val_logits.argmax(axis=1)

        val_f1 = f1_score(
            y_val,
            val_pred,
            average="macro",
            zero_division=0,
        )

        train_loss = running_loss / max(total_items, 1)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_f1, class_weights


# ============================================================
# Training: binary transition detector
# ============================================================

def train_transition_detector(
    model,
    x_train,
    transition_train,
    x_val,
    transition_val,
    device,
):
    positive_count = float(transition_train.sum())
    negative_count = float(len(transition_train) - positive_count)

    # Weighted BCE gives transitions more importance.
    if positive_count > 0:
        pos_weight_value = negative_count / positive_count
    else:
        pos_weight_value = 1.0

    # Avoid excessive weighting.
    pos_weight_value = float(
        np.clip(pos_weight_value, 1.0, 20.0)
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight_value,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    print("\nTransition training distribution:")
    print({
        "non_transition": int(negative_count),
        "transition": int(positive_count),
    })

    print(
        "Transition positive weight:",
        pos_weight_value,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total_items = 0

        for xb, yb in iterate_batches(
            x_train,
            transition_train,
            BATCH_SIZE,
            shuffle=True,
        ):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            running_loss += loss.item() * len(yb)
            total_items += len(yb)

        val_probabilities = predict_transition_probabilities(
            model,
            x_val,
            device,
        )

        val_pred = (
            val_probabilities >= 0.5
        ).astype(np.int64)

        val_f1 = f1_score(
            transition_val,
            val_pred,
            zero_division=0,
        )

        train_loss = running_loss / max(total_items, 1)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val_transition_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_f1, pos_weight_value


# ============================================================
# Hybrid prediction
# ============================================================

def hybrid_predict(
    current_stage,
    next_stage_logits,
    transition_probabilities,
    threshold,
):
    predicted_next_stage = next_stage_logits.argmax(axis=1)

    predicted_transition = (
        transition_probabilities >= threshold
    ).astype(np.int64)

    hybrid_prediction = current_stage.copy()

    transition_mask = predicted_transition == 1
    hybrid_prediction[transition_mask] = (
        predicted_next_stage[transition_mask]
    )

    return (
        hybrid_prediction,
        predicted_transition,
    )


# ============================================================
# Threshold tuning
# ============================================================

def tune_threshold(
    current_stage,
    next_stage,
    next_stage_logits,
    transition_probabilities,
    transition_target,
):
    best_result = None

    print("\nThreshold tuning on validation set:")

    for threshold in THRESHOLDS:
        hybrid_pred, transition_pred = hybrid_predict(
            current_stage=current_stage,
            next_stage_logits=next_stage_logits,
            transition_probabilities=transition_probabilities,
            threshold=threshold,
        )

        stage_metrics = calculate_metrics(
            next_stage,
            hybrid_pred,
        )

        transition_metrics = {
            "transition_accuracy": float(
                accuracy_score(
                    transition_target,
                    transition_pred,
                )
            ),
            "transition_balanced_accuracy": float(
                balanced_accuracy_score(
                    transition_target,
                    transition_pred,
                )
            ),
            "transition_f1": float(
                f1_score(
                    transition_target,
                    transition_pred,
                    zero_division=0,
                )
            ),
        }

        result = {
            "threshold": threshold,
            "stage_metrics": stage_metrics,
            "transition_metrics": transition_metrics,
            "predicted_transition_rate": float(
                transition_pred.mean()
            ),
        }

        print(
            f"threshold={threshold:.2f} | "
            f"stage_macro_f1={stage_metrics['macro_f1']:.4f} | "
            f"stage_balanced_acc={stage_metrics['balanced_accuracy']:.4f} | "
            f"transition_f1={transition_metrics['transition_f1']:.4f} | "
            f"predicted_transition_rate={transition_pred.mean():.4f}"
        )

        if best_result is None:
            best_result = result
        else:
            current_score = (
                stage_metrics["macro_f1"],
                stage_metrics["balanced_accuracy"],
                stage_metrics["accuracy"],
            )

            best_score = (
                best_result["stage_metrics"]["macro_f1"],
                best_result["stage_metrics"]["balanced_accuracy"],
                best_result["stage_metrics"]["accuracy"],
            )

            if current_score > best_score:
                best_result = result

    print("\nBest validation threshold:")
    print(json.dumps(best_result, indent=2))

    return best_result


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    (
        x_train,
        current_train,
        next_train,
        transition_train,
    ) = load_split("train")

    (
        x_val,
        current_val,
        next_val,
        transition_val,
    ) = load_split("val")

    (
        x_test,
        current_test,
        next_test,
        transition_test,
    ) = load_split("test")

    print("\nShapes:")
    print("Train:", x_train.shape)
    print("Val:", x_val.shape)
    print("Test:", x_test.shape)

    print("\nTransition counts:")
    print({
        "train": int(transition_train.sum()),
        "val": int(transition_val.sum()),
        "test": int(transition_test.sum()),
    })

    # --------------------------------------------------------
    # Train next-stage classifier
    # --------------------------------------------------------

    next_stage_model = GRUNextStage(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    next_stage_model, best_next_val_f1, next_class_weights = (
        train_next_stage_model(
            model=next_stage_model,
            x_train=x_train,
            y_train=next_train,
            x_val=x_val,
            y_val=next_val,
            device=device,
        )
    )

    # --------------------------------------------------------
    # Train transition detector
    # --------------------------------------------------------

    transition_model = TransitionDetector(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    transition_model, best_transition_val_f1, pos_weight = (
        train_transition_detector(
            model=transition_model,
            x_train=x_train,
            transition_train=transition_train,
            x_val=x_val,
            transition_val=transition_val,
            device=device,
        )
    )

    # --------------------------------------------------------
    # Generate validation predictions
    # --------------------------------------------------------

    val_next_logits = predict_next_stage_logits(
        next_stage_model,
        x_val,
        device,
    )

    val_transition_probabilities = (
        predict_transition_probabilities(
            transition_model,
            x_val,
            device,
        )
    )

    best_threshold_result = tune_threshold(
        current_stage=current_val,
        next_stage=next_val,
        next_stage_logits=val_next_logits,
        transition_probabilities=val_transition_probabilities,
        transition_target=transition_val,
    )

    best_threshold = best_threshold_result["threshold"]

    # --------------------------------------------------------
    # Generate test predictions
    # --------------------------------------------------------

    test_next_logits = predict_next_stage_logits(
        next_stage_model,
        x_test,
        device,
    )

    test_transition_probabilities = (
        predict_transition_probabilities(
            transition_model,
            x_test,
            device,
        )
    )

    test_hybrid_pred, test_transition_pred = hybrid_predict(
        current_stage=current_test,
        next_stage_logits=test_next_logits,
        transition_probabilities=test_transition_probabilities,
        threshold=best_threshold,
    )

    # --------------------------------------------------------
    # Evaluate test stage predictions
    # --------------------------------------------------------

    test_stage_metrics = evaluate_stage_predictions(
        y_true=next_test,
        y_pred=test_hybrid_pred,
        split_name="Test hybrid",
    )

    test_transition_metrics = evaluate_transition_predictions(
        current_stage=current_test,
        next_stage=next_test,
        predicted_transition=test_transition_pred,
        split_name="Test hybrid",
    )

    actual_transition_rate = float(transition_test.mean())
    predicted_transition_rate = float(
        test_transition_pred.mean()
    )

    print("\nTransition rates:")
    print({
        "actual_transition_rate": actual_transition_rate,
        "predicted_transition_rate": predicted_transition_rate,
    })

    # --------------------------------------------------------
    # Compare with persistence
    # --------------------------------------------------------

    persistence_pred = current_test.copy()

    persistence_metrics = evaluate_stage_predictions(
        y_true=next_test,
        y_pred=persistence_pred,
        split_name="Test persistence",
    )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    checkpoint = {
        "next_stage_model_state_dict": (
            next_stage_model.state_dict()
        ),
        "transition_model_state_dict": (
            transition_model.state_dict()
        ),
        "config": {
            "num_classes": NUM_CLASSES,
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "threshold": best_threshold,
        },
        "next_class_weights": next_class_weights.tolist(),
        "transition_positive_weight": pos_weight,
    }

    checkpoint_path = (
        ARTIFACT_DIR / "gru_hybrid.pt"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    report = {
        "device": str(device),
        "best_next_stage_val_macro_f1": best_next_val_f1,
        "best_transition_val_f1": best_transition_val_f1,
        "selected_threshold": best_threshold,
        "validation_threshold_search": best_threshold_result,
        "test_hybrid_stage_metrics": test_stage_metrics,
        "test_hybrid_transition_metrics": test_transition_metrics,
        "test_persistence_metrics": persistence_metrics,
        "actual_test_transition_rate": actual_transition_rate,
        "predicted_test_transition_rate": predicted_transition_rate,
    }

    report_path = (
        ARTIFACT_DIR / "gru_hybrid_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print("\nSaved:")
    print(checkpoint_path)
    print(report_path)


if __name__ == "__main__":
    main()