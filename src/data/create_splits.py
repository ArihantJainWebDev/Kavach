from pathlib import Path
import json

import numpy as np
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WINDOW_DIR = PROJECT_ROOT / "data" / "processed" / "windows" / "semantic"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_array(name: str) -> np.ndarray:
    path = WINDOW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path)


def chronological_split(array: np.ndarray, train_end: int, val_end: int):
    return (
        array[:train_end],
        array[train_end:val_end],
        array[val_end:],
    )


def main():
    observation_windows = load_array("observation_windows.npy").astype(
        np.float32
    )
    medium_context_windows = load_array("medium_context_windows.npy").astype(
        np.float32
    )
    short_context_windows = load_array("short_context_windows.npy").astype(
        np.float32
    )
    target_features = load_array("target_features.npy").astype(np.float32)
    target_stage_ids = load_array("target_stage_ids.npy").astype(np.int64)

    rollout_stage_ids = load_array("rollout_stage_ids.npy").astype(np.int64)
    rollout_valid_mask = load_array("rollout_valid_mask.npy").astype(bool)

    n_samples = len(target_stage_ids)

    if not all(
        len(array) == n_samples
        for array in [
            observation_windows,
            medium_context_windows,
            short_context_windows,
            target_features,
            rollout_stage_ids,
            rollout_valid_mask,
        ]
    ):
        raise ValueError("All arrays must contain the same number of samples.")

    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    print(f"Total samples: {n_samples}")
    print(f"Train samples: {train_end}")
    print(f"Validation samples: {val_end - train_end}")
    print(f"Test samples: {n_samples - val_end}")

    arrays = {
        "observation_windows": observation_windows,
        "medium_context_windows": medium_context_windows,
        "short_context_windows": short_context_windows,
        "target_features": target_features,
        "target_stage_ids": target_stage_ids,
        "rollout_stage_ids": rollout_stage_ids,
        "rollout_valid_mask": rollout_valid_mask,
    }

    split_data = {}

    for name, array in arrays.items():
        train, val, test = chronological_split(
            array,
            train_end,
            val_end,
        )

        split_data[name] = {
            "train": train,
            "val": val,
            "test": test,
        }

    # Fit the scaler only on the training observations.
    #
    # The observation windows have shape:
    # (samples, sequence_length, features)
    train_observations = split_data["observation_windows"]["train"]

    n_features = train_observations.shape[-1]

    scaler = StandardScaler()
    scaler.fit(
        train_observations.reshape(-1, n_features)
    )

    for split_name in ["train", "val", "test"]:
        x = split_data["observation_windows"][split_name]

        split_data["observation_windows"][split_name] = scaler.transform(
            x.reshape(-1, n_features)
        ).reshape(x.shape).astype(np.float32)

    for split_name in ["train", "val", "test"]:
        x = split_data["medium_context_windows"][split_name]

        split_data["medium_context_windows"][split_name] = scaler.transform(
            x.reshape(-1, n_features)
        ).reshape(x.shape).astype(np.float32)

    for split_name in ["train", "val", "test"]:
        x = split_data["short_context_windows"][split_name]

        split_data["short_context_windows"][split_name] = scaler.transform(
            x.reshape(-1, n_features)
        ).reshape(x.shape).astype(np.float32)

    for split_name in ["train", "val", "test"]:
        x = split_data["target_features"][split_name]

        split_data["target_features"][split_name] = scaler.transform(
            x
        ).astype(np.float32)

    # Save split arrays.
    for name, split_dict in split_data.items():
        for split_name, array in split_dict.items():
            output_path = OUTPUT_DIR / f"{name}_{split_name}.npy"
            np.save(output_path, array)
            print(f"Saved: {output_path} | shape={array.shape}")

    scaler_path = OUTPUT_DIR / "feature_scaler.npz"

    np.savez(
        scaler_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
        n_features_in=np.array([scaler.n_features_in_]),
    )

    metadata = {
        "total_samples": n_samples,
        "train_samples": train_end,
        "validation_samples": val_end - train_end,
        "test_samples": n_samples - val_end,
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "split_strategy": "chronological",
        "scaler_fit_on": "training_observation_windows_only",
        "n_features": n_features,
    }

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nChronological split creation completed successfully.")


if __name__ == "__main__":
    main()