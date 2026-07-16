"""
component.py

Sky component model: combines a latent spatial field with a linear mixing matrix
and activation to produce one component's physical flux cube.

s = activation(A @ u + offset)

where u are the latent field values (from any *Field base), A is the mixing
matrix describing cross-wavelength correlations, and the activation maps to
physical flux (e.g. exp for positivity).

A sky is built by summing several SkyComponents (e.g. diffuse + point sources);
each component carries its own mixing matrix and must use a distinct `prefix` so
the parameter domains stay disjoint.
"""

# === Imports ======================================================================================

from typing import Any

import jax
import jax.numpy as jnp
import nifty8.re as jft

from astroprism.models.sky.diffuse import DiffuseField
from astroprism.utils.priors import build_prior
from astroprism.utils.config import get_defaults

# === Main =========================================================================================

ACTIVATIONS = {
    "exp": jnp.exp,
    "softplus": lambda x: jnp.log1p(jnp.exp(x)),
    "sigmoid": jax.nn.sigmoid,
    "identity": lambda x: x,
    "relu": jax.nn.relu,
    "square": lambda x: x**2,
}


class SkyComponent(jft.Model):
    """
    One component of the sky: a latent spatial field mixed across channels.

    s = activation(A @ u + offset)

    Parameters
    ----------
    field : jft.Model
        Latent spatial base (e.g. DiffuseField or PointSourceField). Must expose
        `.domain`, `.init`, `.n_channels`, `.shape`, `.distances` and return a
        `(n_channels, ny, nx)` array when called.
    mixing_mode : str
        "full" for dense mixing matrix, "cholesky" for Cholesky parameterization.
    mixing_matrix : dict
        Prior config for the mixing matrix (used for both modes).
    mixing_offset : dict
        Prior config for the per-channel offset.
    activation : str
        Activation function name (exp, softplus, sigmoid, identity, relu, square).
    prefix : str
        Prepended to this component's parameter names so multiple components
        summed into a SkyModel keep disjoint domains (e.g. "diffuse", "point").
    """

    def __init__(
        self,
        field: jft.Model,
        mixing_mode: str = None,
        mixing_matrix: dict = None,
        mixing_offset: dict = None,
        activation: str = None,
        prefix: str = "",
    ):
        s = get_defaults()["component"]
        activation    = activation    if activation    is not None else s.get("activation", "exp")
        mixing_mode   = mixing_mode   if mixing_mode   is not None else s.get("mixing_mode", "full")
        mixing_matrix = mixing_matrix or s["mixing_matrix"]
        mixing_offset = mixing_offset or s["mixing_offset"]

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(ACTIVATIONS.keys())}")
        self.activation = ACTIVATIONS[activation]
        self.activation_name = activation
        self.field = field
        self.mixing_mode = mixing_mode
        self.prefix = prefix
        domain = self.field.domain
        init = self.field.init
        n_channels = self.field.n_channels

        if self.mixing_mode == "full":
            domain, init = self._build_full_mixing(n_channels, domain, init, mixing_matrix)
        elif self.mixing_mode == "cholesky":
            domain, init = self._build_cholesky_mixing(n_channels, domain, init, mixing_matrix)
        else:
            raise ValueError(f"Unknown mixing_mode: {mixing_mode}. Choose 'full' or 'cholesky'.")

        self.mixing_offset = build_prior(self._name("mixing_offset"), mixing_offset, (n_channels,))
        domain = domain | self.mixing_offset.domain
        init = init | self.mixing_offset.init

        super().__init__(domain=domain, init=init)

    @property
    def n_channels(self) -> int:
        return self.field.n_channels

    @property
    def shape(self) -> tuple[int, ...]:
        return self.field.shape

    @property
    def distances(self) -> tuple[float, ...]:
        return self.field.distances

    def __call__(self, x: dict[str, Any]) -> jnp.ndarray:
        base_outputs = self.field({k: x[k] for k in self.field.domain.keys()})

        if self.mixing_mode == "full":
            mixing_matrix = self.mixing_matrix(x)
        elif self.mixing_mode == "cholesky":
            diag = self.mixing_diag(x)
            off_diag = self.mixing_off_diag(x)
            mixing_matrix = _assemble_cholesky_matrix(diag, off_diag)

        mixture_output = jnp.tensordot(mixing_matrix, base_outputs, axes=(1, 0))
        return self.activation(mixture_output + self.mixing_offset(x)[:, None, None])

    def sample(self, key: int | jax.Array) -> jnp.ndarray:
        """Generate a random sample from the prior."""
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        params = self.init(key)
        return jnp.array(self(params))

    def _name(self, base: str) -> str:
        """Prefix a parameter name to keep component domains disjoint."""
        return f"{self.prefix}_{base}" if self.prefix else base

    def _build_full_mixing(self, n_channels, domain, init, mixing):
        self.mixing_matrix = build_prior(self._name("mixing_matrix"), mixing, (n_channels, n_channels))
        return domain | self.mixing_matrix.domain, init | self.mixing_matrix.init

    def _build_cholesky_mixing(self, n_channels, domain, init, mixing):
        self.mixing_diag     = build_prior(self._name("mixing_diag"),     mixing, (n_channels,))
        self.mixing_off_diag = build_prior(self._name("mixing_off_diag"), mixing, (_n_triangular_lower(n_channels),))
        return (
            domain | self.mixing_diag.domain | self.mixing_off_diag.domain,
            init | self.mixing_diag.init | self.mixing_off_diag.init,
        )


# === Utilities ====================================================================================

def _assemble_cholesky_matrix(diag, off_diag):
    """Assemble a Cholesky matrix from a diagonal and off-diagonal vector."""
    n_channels = len(diag)
    L = jnp.zeros((n_channels, n_channels))
    L = L.at[jnp.diag_indices(n_channels)].set(1.0)
    tril_indices = jnp.tril_indices(n_channels, k=-1)
    L = L.at[tril_indices].set(off_diag)
    L = L * (jnp.exp(diag))[:, None]
    return L


def _n_triangular_lower(n):
    """Number of elements in the lower triangular part of a square matrix."""
    return (n * (n - 1)) // 2
