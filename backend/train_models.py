import json
from pathlib import Path

from training.pipeline import DATASET_PATH, train_all


def main() -> None:
    metrics = train_all(DATASET_PATH)
    print("\nTraining complete")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved models: {Path(__file__).resolve().parent / 'saved_models'}")


if __name__ == "__main__":
    main()
