"""Unified entry point for Natural Memory v2 local production workflows.

Examples:
  python -m V2_dpskw.natural_memory_app chat --model-path ...
  python -m V2_dpskw.natural_memory_app serve --port 8765
  python -m V2_dpskw.natural_memory_app build-dataset
  python -m V2_dpskw.natural_memory_app train-policy --steps 240
  python -m V2_dpskw.natural_memory_app stress --rounds 40
  python -m V2_dpskw.natural_memory_app make-mega-validation
  python -m V2_dpskw.natural_memory_app benchmark-kv --limit 32
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "chat",
            "serve",
            "build-dataset",
            "train-policy",
            "stress",
            "make-mega-validation",
            "benchmark-kv",
        ),
        help="workflow to run; remaining arguments are passed to that workflow",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.command == "chat":
        from .stream_chat_qwen_memory import main as run
    elif args.command == "serve":
        from .natural_memory_service import main as run
    elif args.command == "build-dataset":
        from .build_production_memory_dataset import main as run
    elif args.command == "train-policy":
        from .train_production_memory_policy import main as run
    elif args.command == "stress":
        from .stress_test_natural_memory import main as run
    elif args.command == "make-mega-validation":
        from .make_mega_memory_validation import main as run
    else:
        from .benchmark_memory_vs_full_kv import main as run
    run()


if __name__ == "__main__":
    main()
