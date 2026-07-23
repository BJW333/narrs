"""
Run the full validation battery.

    python examples/run_battery.py               # both landscapes + aggregator study
    python examples/run_battery.py --quick       # skip the aggregator study

Prints an overfitting report card per landscape and writes the labelled
(search features -> what happened out of sample) records that a self-tuning or
meta-learning loop would train on.

Requires nothing beyond the standard library. Install ``optuna`` and ``cma`` to
enable two of the three comparison baselines; without them the battery still
runs and simply reports fewer methods.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from narrs.benchmarks import run_all
from narrs.benchmarks.aggregator_study import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description="NARRS validation battery")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="skip the aggregator study")
    parser.add_argument(
        "--flywheel",
        default="flywheel_records.jsonl",
        help="where to append the labelled records (blank to disable)",
    )
    args = parser.parse_args()

    flywheel = args.flywheel or None
    if flywheel and os.path.exists(flywheel):
        os.remove(flywheel)

    run_all(seed=args.seed, flywheel_path=flywheel)

    if not args.quick:
        run_study(seed=args.seed)

    if flywheel and os.path.exists(flywheel):
        with open(flywheel) as handle:
            n = sum(1 for _ in handle)
        print(f"\nWrote {n} labelled records -> {flywheel}")
        print(
            "Each record pairs the search-landscape features NARRS could observe "
            "before knowing the answer with what actually happened out of sample. "
            "That is the calibration data a self-tuning loop needs, and the corpus "
            "a meta-learning layer would learn priors from."
        )


if __name__ == "__main__":
    main()
