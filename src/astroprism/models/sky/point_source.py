"""
point_source.py

Point-source latent spatial field model using a per-pixel inverse-gamma prior.

Unlike DiffuseField (a smooth, spatially-correlated Gaussian process), point
sources use a sparse, heavy-tailed inverse-gamma prior applied independently per
pixel: most pixels sit near zero while a few become very bright, which is what
lets isolated sources (stars, AGN) form as sharp spikes rather than smooth
structure.

Like DiffuseField, the field is returned in log-space so that the downstream
SkyComponent activation (exp) recovers physical flux — mirroring J-UBIK's
`jnp.log(invgamma_prior(...))` convention (see jubik/sky_models.py).

The sparse base is factored into `_build_sparse_field` so a future
SemiResolvedField can reuse it (sparse base convolved with a small Moffat).
"""

# === Imports ======================================================================================

from typing import Any

import jax
import jax.numpy as jnp
import nifty8.re as jft
from astroprism.utils.config import get_defaults

# === Main =========================================================================================

class PointSourceField(jft.Model):
    """A stack of independent per-pixel inverse-gamma fields, one per channel."""

    def __init__(
        self,
        n_channels: int,
        shape: tuple[int, ...],
        distances: tuple[float, ...],
        alpha: float = None,
        q: float = None,
        name: str = "points",
        prefix: str = "",
    ):
        # point_source is optional in the default config, so fall back gracefully.
        ps = get_defaults().get("point_source", {}).get("spatial", {})
        alpha = alpha if alpha is not None else ps.get("alpha", 1.0)
        q     = q     if q     is not None else ps.get("q", 0.1)

        key = f"{prefix}_{name}" if prefix else name
        self._key = key
        self._invgamma = self._build_sparse_field(alpha, q)
        self._n_channels = n_channels
        self._shape = shape
        self._distances = distances

        domain = {key: jft.ShapeWithDtype((n_channels,) + tuple(shape), jnp.float64)}
        super().__init__(domain=domain, white_init=True)

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def distances(self) -> tuple[float, ...]:
        return self._distances

    def __call__(self, x: dict[str, Any]) -> jnp.ndarray:
        # Inverse-gamma flux in log-space, so SkyComponent's exp activation
        # recovers the sparse flux (matches DiffuseField's pre-activation space).
        return jnp.log(self._invgamma(x[self._key]))

    def sample(self, key: int | jax.Array) -> jnp.ndarray:
        """Generate a random sample from the prior."""
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        params = self.init(key)
        return jnp.array(self(params))

    @staticmethod
    def _build_sparse_field(alpha, q):
        """
        Build the per-pixel inverse-gamma transform.

        Returns an elementwise callable mapping a standard-normal latent array to
        inverse-gamma distributed values (shape preserved). Reusable as the sparse
        base for other point-like components (e.g. semi-resolved sources).
        """
        return jft.invgamma_prior(a=alpha, scale=q)
