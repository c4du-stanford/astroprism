"""
sky.py

Sky model: the sum of one or more named sky components (e.g. diffuse + point
sources). Each component is a SkyComponent producing a physical flux cube; the
sky is their elementwise sum at flux level.

The sum must happen *after* each component's activation (exp), since the model is
additive in flux: total_flux = diffuse_flux + point_flux (not exp of a summed
latent). Components are stored by name so individual components can be evaluated
separately for inspection/plotting after fitting, and so further components
(e.g. semi-resolved sources) drop in as additional named entries.
"""

# === Imports ======================================================================================

from typing import Any

import jax
import jax.numpy as jnp
import nifty8.re as jft

# === Main =========================================================================================

class SkyModel(jft.Model):
    """
    Sum of named sky components.

    Parameters
    ----------
    components : dict[str, jft.Model]
        Named components (e.g. {"diffuse": ..., "point": ...}). Each must return a
        `(n_channels, ny, nx)` flux cube and have a domain disjoint from the others
        (ensured by per-component prefixes in SkyComponent).
    """

    def __init__(self, components: dict[str, jft.Model]):
        if not components:
            raise ValueError("SkyModel needs at least one component.")
        self.components = dict(components)

        domain: dict = {}
        init_parts = []
        for name, comp in self.components.items():
            overlap = set(domain) & set(comp.domain)
            if overlap:
                raise ValueError(
                    f"Component '{name}' has overlapping domain keys {sorted(overlap)}; "
                    "components must use distinct prefixes to keep domains disjoint."
                )
            domain |= comp.domain
            init_parts.append(comp.init)

        def init(key):
            params: dict = {}
            for part in init_parts:
                params |= part(key)
            return params

        super().__init__(domain=domain, init=init)

    @property
    def component_names(self) -> list[str]:
        return list(self.components.keys())

    @property
    def n_channels(self) -> int:
        return next(iter(self.components.values())).n_channels

    @property
    def shape(self) -> tuple[int, ...]:
        return next(iter(self.components.values())).shape

    @property
    def distances(self) -> tuple[float, ...]:
        return next(iter(self.components.values())).distances

    def __call__(self, x: dict[str, Any]) -> jnp.ndarray:
        """Total sky flux: sum of all component flux cubes."""
        comps = iter(self.components.values())
        total = next(comps)(x)
        for comp in comps:
            total = total + comp(x)
        return total

    def evaluate_components(self, x: dict[str, Any]) -> dict[str, jnp.ndarray]:
        """Per-component flux cubes, keyed by name (for inspection/plotting)."""
        return {name: comp(x) for name, comp in self.components.items()}

    def sample(self, key: int | jax.Array) -> jnp.ndarray:
        """Generate a random sample of the total sky from the prior."""
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        params = self.init(key)
        return jnp.array(self(params))
