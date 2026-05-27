"""
results.py

Load and analyze results from a completed astroprism inference run.
"""

# === Imports ======================================================================================

import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

from astroprism.models.field import FieldModel
from astroprism.models.signal import SignalModel
from astroprism.models.response import InstrumentResponse
from astroprism.models.noise import NoiseModel
from astroprism.models.forward import ForwardModel
from astroprism.io.dataset import load_dataset

# === Main =========================================================================================

class PosteriorResult:
    """
    Load and analyze results from a completed astroprism run.

    Models and samples are loaded lazily and cached. Predictions are computed
    in a single pass through the forward model to avoid redundant work.

    Parameters
    ----------
    run_dir : str or Path
        Path to the output directory containing config.yaml and last.pkl.

    Examples
    --------
    result = PosteriorResult("output/jwst_miri_tutorial")

    # Signal only (no dataset needed)
    predictions = result.predict()
    signal_mean = predictions["signal_mean"]

    # Full pipeline (needs dataset)
    predictions = result.predict(quantities=["signal", "response", "noise_std"])
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)

        with open(self.run_dir / "config.yaml") as f:
            self._config = yaml.safe_load(f)

        files_path = self.run_dir / "files_used.yaml"
        if files_path.exists():
            with open(files_path) as f:
                self._files_used = yaml.safe_load(f)
        else:
            self._files_used = None

        # Lazy caches
        self._samples = None
        self._state = None
        self._dataset = None
        self._signal_model = None
        self._response_model = None
        self._noise_model = None

    # === Properties ==============================================================================

    @property
    def config(self) -> dict:
        """The full config used for this run."""
        return self._config

    @property
    def derived(self) -> dict:
        """Derived values (n_channels, signal_shape, distances, etc.)."""
        return self._config["_derived"]

    @property
    def samples(self):
        """Posterior samples (lazy loaded, cached)."""
        if self._samples is None:
            with open(self.run_dir / "last.pkl", "rb") as f:
                self._samples, self._state = pickle.load(f)
        return self._samples

    @property
    def dataset(self):
        """Original dataset (lazy loaded from files_used.yaml, cached)."""
        if self._dataset is None:
            if self._files_used is None:
                raise FileNotFoundError("No files_used.yaml — cannot reload dataset.")
            self._dataset = load_dataset(
                path=self._files_used["data_path"],
                instrument=self._files_used["instrument"],
                extension=self._files_used.get("extension", "fits"),
            )
        return self._dataset

    @property
    def signal_model(self) -> SignalModel:
        """Signal model (lazy built, cached)."""
        if self._signal_model is None:
            d = self.derived
            field = FieldModel(
                n_channels=d["n_channels"],
                shape=tuple(d["signal_shape"]),
                distances=tuple(d["distances"]),
            )
            self._signal_model = SignalModel(field)
        return self._signal_model

    @property
    def response_model(self) -> InstrumentResponse:
        """Response model (lazy built, cached). Triggers dataset load."""
        if self._response_model is None:
            d = self.derived
            ds = self.dataset
            self._response_model = InstrumentResponse(
                dataset=ds,
                signal_wcs=ds.wcs[d["ref_idx"]],
                signal_shape=tuple(d["signal_shape"]),
            )
        return self._response_model

    @property
    def noise_model(self) -> NoiseModel:
        """Noise model (lazy built, cached)."""
        if self._noise_model is None:
            self._noise_model = NoiseModel(n_channels=self.derived["n_channels"])
        return self._noise_model

    # === Predictions =============================================================================

    def predict(self, quantities=None, dataset=None):
        """
        Compute predictions in a single pass through the forward model.

        Parameters
        ----------
        quantities : list of str, optional
            What to compute. Options: "signal", "response", "noise_std".
            Default: ["signal"].
        dataset : BaseDataset, optional
            Dataset for response/noise predictions. Loaded from files_used.yaml
            if not provided.

        Returns
        -------
        dict with keys:
            "signal"    : list of (n_channels, ny, nx) arrays, one per sample
            "response"  : list of [ch0_array, ch1_array, ...], one per sample
            "noise_std" : list of [ch0_array, ch1_array, ...], one per sample
            "signal_mean" : (n_channels, ny, nx) mean across samples
            "signal_std"  : (n_channels, ny, nx) std across samples
        Only requested quantities (and their means/stds) are included.
        """
        if quantities is None:
            quantities = ["signal"]

        need_response = "response" in quantities or "noise_std" in quantities

        results = {q: [] for q in quantities}

        for s in self.samples:
            x = s.tree

            if "signal" in quantities or need_response:
                sig = self.signal_model(x)

            if "signal" in quantities:
                results["signal"].append(jnp.array(sig))

            if need_response:
                resp = self.response_model(x, sig)
                if "response" in quantities:
                    results["response"].append(resp)
                if "noise_std" in quantities:
                    results["noise_std"].append(self.noise_model(x, resp))

        # Compute mean/std for signal
        if "signal" in quantities:
            stacked = jnp.stack(results["signal"])
            results["signal_mean"] = jnp.mean(stacked, axis=0)
            results["signal_std"] = jnp.std(stacked, axis=0)

        return results
