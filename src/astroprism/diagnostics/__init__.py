"""Diagnostics: plotting and metrics for posterior analysis."""

from astroprism.diagnostics.plots import plot_channel_grid, plot_parity_grid, plot_residual_histograms, plot_power_spectrum, plot_sky_components
from astroprism.diagnostics.metrics import noise_weighted_residuals, uncertainty_weighted_residuals, reduced_chi_squared, power_spectrum

__all__ = [
    "plot_channel_grid",
    "plot_parity_grid",
    "plot_residual_histograms",
    "plot_power_spectrum",
    "plot_sky_components",
    "noise_weighted_residuals",
    "uncertainty_weighted_residuals",
    "reduced_chi_squared",
    "power_spectrum",
]
