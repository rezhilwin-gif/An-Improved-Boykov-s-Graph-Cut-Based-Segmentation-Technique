"""
Top-level orchestrator for the IBGC-CRF-SPSST pipeline.

Runs, in order:
  1. src/inspect_data.py     - environment + dataset + ground-truth color checks
  2. src/train.py            - trains VGG16-FCN8, saves checkpoints/best_model.pt
  3. src/tune_hyperparams.py - sweeps graph-cut/CRF params on validation data
  4. src/final_eval.py       - full held-out evaluation, metrics to outputs/
  5. src/visualize_preds.py  - saves stage-by-stage figures to graphs/

Usage:
    python run_pipeline.py                 # run every stage
    python run_pipeline.py --skip-train     # reuse an existing checkpoint
    python run_pipeline.py --only inspect,train

All stages read/write through the shared paths defined in src/config.py, so
they can also be run individually, e.g. `python src/train.py`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

STAGES = ["inspect", "train", "tune", "eval", "visualize"]


def run_stage(name):
    print(f"\n{'=' * 70}\nSTAGE: {name}\n{'=' * 70}")
    if name == "inspect":
        import inspect_data

        inspect_data.main()
    elif name == "train":
        import train

        train.main()
    elif name == "tune":
        import tune_hyperparams

        tune_hyperparams.main()
    elif name == "eval":
        import final_eval

        final_eval.main()
    elif name == "visualize":
        import visualize_preds
        import sys as _sys

        _sys.argv = ["visualize_preds.py", "--n", "1"]
        visualize_preds.main()
    else:
        raise ValueError(f"Unknown stage: {name}")


def main():
    parser = argparse.ArgumentParser(description="Run the IBGC-CRF-SPSST pipeline end to end.")
    parser.add_argument("--only", type=str, default=None, help="comma-separated subset of stages to run: " + ",".join(STAGES))
    parser.add_argument("--skip-train", action="store_true", help="skip training (reuse checkpoints/best_model.pt)")
    args = parser.parse_args()

    if args.only:
        stages = [s.strip() for s in args.only.split(",")]
        for s in stages:
            if s not in STAGES:
                raise ValueError(f"Unknown stage '{s}'. Valid stages: {STAGES}")
    else:
        stages = list(STAGES)
        if args.skip_train:
            stages.remove("train")

    for stage in stages:
        run_stage(stage)

    print("\nPipeline complete.")
    print("  Checkpoints -> checkpoints/")
    print("  Graphs      -> graphs/")
    print("  Outputs     -> outputs/")


if __name__ == "__main__":
    main()
