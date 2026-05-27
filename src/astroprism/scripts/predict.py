"""
predict.py

CLI entry point for generating predictions from a completed astroprism run.

Usage
-----
astroprism-predict --run-dir output/run_001
astroprism-predict --run-dir output/run_001 --quantities signal response noise_std
"""

# === Imports ======================================================================================

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import numpy as np

from astroprism.io.results import PosteriorResult

# === Main =========================================================================================

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Generate predictions from a completed astroprism run.")
        parser.add_argument("--run-dir", required=True, metavar="PATH", help="Path to run output directory")
        parser.add_argument(
            "--quantities", nargs="+", default=["signal"],
            choices=["signal", "response", "noise_std"],
            help="Quantities to predict",
        )
        parser.add_argument("--output-dir", default=None, metavar="PATH", help="Output directory (default: run-dir/predictions/)")
        args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.run_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    # Load results and predict
    result = PosteriorResult(args.run_dir)
    print(f"Loaded run: {args.run_dir}")
    print(f"  n_channels: {result.derived['n_channels']}")
    print(f"  signal_shape: {result.derived['signal_shape']}")
    print(f"  channel_keys: {result.derived['channel_keys']}")

    predictions = result.predict(quantities=args.quantities)

    # Save results
    for quantity in args.quantities:
        if quantity == "signal":
            # Signal is (n_channels, ny, nx) per sample — stack and save
            np.savez(
                os.path.join(output_dir, "signal.npz"),
                mean=np.asarray(predictions["signal_mean"]),
                std=np.asarray(predictions["signal_std"]),
                samples=np.asarray(np.stack([np.asarray(s) for s in predictions["signal"]])),
            )
            print(f"  Saved signal.npz (mean, std, samples)")

        elif quantity == "response":
            # Response is list of per-channel arrays per sample — save mean
            n_channels = result.derived["n_channels"]
            response_mean = {}
            for ch in range(n_channels):
                ch_arrays = [np.asarray(sample[ch]) for sample in predictions["response"]]
                response_mean[f"ch{ch}_mean"] = np.mean(ch_arrays, axis=0)
            np.savez(os.path.join(output_dir, "response.npz"), **response_mean)
            print(f"  Saved response.npz (per-channel means)")

        elif quantity == "noise_std":
            n_channels = result.derived["n_channels"]
            noise_mean = {}
            for ch in range(n_channels):
                ch_arrays = [np.asarray(sample[ch]) for sample in predictions["noise_std"]]
                noise_mean[f"ch{ch}_mean"] = np.mean(ch_arrays, axis=0)
            np.savez(os.path.join(output_dir, "noise_std.npz"), **noise_mean)
            print(f"  Saved noise_std.npz (per-channel means)")

    print(f"Done. Predictions saved to: {output_dir}")
