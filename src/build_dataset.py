from pathlib import Path

from .state_builder import build_states_from_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "unraveled"
    / "data"
    / "network-flows"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "states"
)


if __name__ == "__main__":
    build_states_from_dataset(
        dataset_root=DATASET_ROOT,
        output_dir=OUTPUT_DIR,
        window="60s",
        chunksize=50_000,
    )