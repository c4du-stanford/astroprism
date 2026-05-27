"""
cli.py

Main CLI entry point for astroprism.

Usage
-----
astroprism run --config configs/my_run.yaml
astroprism predict --run-dir output/run_001
"""

# === Imports ======================================================================================

import jax
jax.config.update("jax_enable_x64", True)

import argparse

from astroprism.scripts.run import main as run_main
from astroprism.scripts.predict import main as predict_main

# === Main =========================================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="astroprism",
        description="Bayesian inference for multi-channel astronomical imaging.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run inference pipeline")
    run_parser.add_argument("--config", required=True, metavar="PATH", help="Path to config YAML")

    # --- predict ---
    predict_parser = subparsers.add_parser("predict", help="Generate predictions from a completed run")
    predict_parser.add_argument("--run-dir", required=True, metavar="PATH", help="Path to run output directory")
    predict_parser.add_argument(
        "--quantities", nargs="+", default=["signal"],
        choices=["signal", "response", "noise_std"],
        help="Quantities to predict (default: signal)",
    )
    predict_parser.add_argument("--output-dir", default=None, metavar="PATH", help="Output directory (default: run-dir/predictions/)")

    args = parser.parse_args()

    if args.command == "run":
        run_main(args)
    elif args.command == "predict":
        predict_main(args)
