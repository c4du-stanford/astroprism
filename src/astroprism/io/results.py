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

from astroprism.models.sky import DiffuseField, PointSourceField, SkyComponent, SkyModel
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
    def sky_model(self) -> SkyModel:
        """Sky model (lazy built, cached). Mirrors the components built in run.py."""
        if self._signal_model is None:
            d = self.derived
            kw = dict(
                n_channels=d["n_channels"],
                shape=tuple(d["signal_shape"]),
                distances=tuple(d["distances"]),
            )
            components = {
                "diffuse": SkyComponent(DiffuseField(**kw), prefix="diffuse"),
            }
            if "point_source" in self._config:
                components["point"] = SkyComponent(PointSourceField(**kw), prefix="point")
            self._signal_model = SkyModel(components)
        return self._signal_model

    # Back-compat alias (pre-rename name).
    @property
    def signal_model(self) -> SkyModel:
        """Deprecated alias for `sky_model`."""
        return self.sky_model

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
            What to compute. Options: "signal", "components", "response",
            "noise_std". Default: ["signal"].
            - "signal"     : the total (summed) sky flux.
            - "components" : per-component flux cubes (e.g. diffuse, point),
                             keyed by component name. Use to inspect each sky
                             component separately after fitting.
        dataset : BaseDataset, optional
            Dataset for response/noise predictions. Loaded from files_used.yaml
            if not provided.

        Returns
        -------
        dict with keys:
            "signal"     : list of (n_channels, ny, nx) arrays, one per sample
            "components" : dict {name: list of (n_channels, ny, nx) arrays}, plus
                           "<name>_mean"/"<name>_std" per component
            "response"   : list of [ch0_array, ch1_array, ...], one per sample
            "noise_std"  : list of [ch0_array, ch1_array, ...], one per sample
            "signal_mean" : (n_channels, ny, nx) mean across samples
            "signal_std"  : (n_channels, ny, nx) std across samples
        Only requested quantities (and their means/stds) are included.
        """
        if quantities is None:
            quantities = ["signal"]

        need_response = "response" in quantities or "noise_std" in quantities
        need_components = "components" in quantities
        component_names = self.sky_model.component_names

        results = {q: [] for q in quantities if q != "components"}
        if need_components:
            results["components"] = {name: [] for name in component_names}

        for s in self.samples:
            x = s.tree

            if need_components:
                comps = self.sky_model.evaluate_components(x)
                for name, cube in comps.items():
                    results["components"][name].append(jnp.array(cube))

            if "signal" in quantities or need_response:
                sig = self.sky_model(x)

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

        # Compute mean/std per component
        if need_components:
            for name in component_names:
                stacked = jnp.stack(results["components"][name])
                results["components"][f"{name}_mean"] = jnp.mean(stacked, axis=0)
                results["components"][f"{name}_std"] = jnp.std(stacked, axis=0)

        return results
