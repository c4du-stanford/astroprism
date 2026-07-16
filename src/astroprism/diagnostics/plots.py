"""
plots.py

Plotting utilities for multi-channel astronomical imaging diagnostics.
All plot functions accept lists of 2D arrays (ragged channel shapes).
"""

# === Imports ======================================================================================

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

# === Main =========================================================================================

def plot_channel_grid(
    images,
    channel_keys=None,
    wcs=None,
    scale="log",
    vmin=None,
    vmax=None,
    cmap="viridis",
    title=None,
    cbar_label="Flux",
    figsize=None,
    n_rows=None,
    n_cols=None,
    axes=None,
    savefig=None,
    show=True,
):
    """
    Plot a grid of per-channel images.

    Parameters
    ----------
    images : list of 2D arrays
        One image per channel (ragged shapes OK).
    channel_keys : list of str, optional
        Labels for each channel.
    wcs : WCS, optional
        WCS projection applied to all subplots.
    scale : str
        "log" or "linear".
    vmin, vmax : float, optional
        Color scale limits. Auto-computed if None.
    cmap : str
        Matplotlib colormap.
    title : str, optional
        Figure title.
    cbar_label : str
        Colorbar label.
    savefig : str, optional
        Path to save the figure.
    show : bool
        Call plt.show() at the end.

    Returns
    -------
    fig, axes
    """
    n_channels = len(images)
    fig, axes = _setup_grid(n_channels, wcs, figsize, n_rows, n_cols, axes)

    # Auto vmin/vmax
    if vmin is None:
        vmin = min(float(np.nanmin(np.asarray(img)[np.asarray(img) > 0])) if np.any(np.asarray(img) > 0) else 1e-3 for img in images)
    if vmax is None:
        vmax = max(float(np.nanmax(np.asarray(img))) for img in images)

    # Normalization
    if scale == "log":
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif scale == "linear":
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    for idx, img in enumerate(images):
        ax = axes[idx]
        im = ax.imshow(np.asarray(img), origin="lower", norm=norm, cmap=cmap, interpolation="none")
        label = channel_keys[idx] if channel_keys and idx < len(channel_keys) else f"Channel {idx}"
        ax.set_title(label)
        ax.axis("off")

    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.04, 0.8, 0.025])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label=cbar_label)

    if title:
        fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    _finish(fig, savefig, show)
    return fig, axes


def plot_sky_components(
    result,
    channel=0,
    dataset=None,
    scale="log",
    cmap="viridis",
    figsize=None,
    savefig=None,
    show=True,
):
    """
    Visualize the sky decomposition for a single channel after fitting:
    each sky component separately (e.g. GP mixture, point sources), then the
    summed sky, then the convolved+noisy prediction, then the real data.

    Generalizes the notebook "sky vs observed vs data" plot to show the
    individual components that sum into the sky.

    Parameters
    ----------
    result : PosteriorResult
        A loaded run. Used to compute per-component, summed-sky, and response
        predictions via `result.predict`.
    channel : int
        Which channel (wavelength) to display.
    dataset : BaseDataset, optional
        Dataset for the observed-data panel and the response prediction. Loaded
        from the run's files_used.yaml if not provided.
    scale : str
        "log" or "linear" color scale.
    cmap : str
        Matplotlib colormap.
    savefig : str, optional
        Path to save the figure.
    show : bool
        Call plt.show() at the end.

    Returns
    -------
    fig, axes
    """
    ds = dataset if dataset is not None else result.dataset
    preds = result.predict(
        quantities=["components", "signal", "response"], dataset=ds
    )

    comp_block = preds["components"]
    comp_names = [k for k in comp_block if not (k.endswith("_mean") or k.endswith("_std"))]

    # Build the ordered list of (label, 2D image) panels for this channel.
    panels = []
    for name in comp_names:
        panels.append((f"{name} (mean)", np.asarray(comp_block[f"{name}_mean"])[channel]))
    panels.append(("sky = Σ components", np.asarray(preds["signal_mean"])[channel]))

    # Convolved + noisy prediction (response mean across samples) and the data.
    resp_samples = preds["response"]
    resp_mean_ch = np.mean([np.asarray(s[channel]) for s in resp_samples], axis=0)
    panels.append(("convolved (response)", resp_mean_ch))
    panels.append(("data", np.asarray(ds.data[channel])))

    images = [p[1] for p in panels]
    labels = [p[0] for p in panels]

    fig, axes = _setup_grid(len(panels), figsize=figsize, n_rows=1)

    # Shared color scale across the sky/component/response panels (positive flux);
    # the data panel shares it too so they are visually comparable.
    pos = [img[img > 0] for img in images if np.any(img > 0)]
    vmin = min(float(p.min()) for p in pos) if pos else 1e-3
    vmax = max(float(np.nanmax(img)) for img in images)
    if scale == "log":
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif scale == "linear":
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    for idx, (label, img) in enumerate(zip(labels, images)):
        ax = axes[idx]
        im = ax.imshow(img, origin="lower", norm=norm, cmap=cmap, interpolation="none")
        ax.set_title(label)
        ax.axis("off")

    cbar_ax = fig.add_axes([0.1, 0.04, 0.8, 0.025])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Flux")

    ch_key = None
    if getattr(ds, "channel_keys", None) is not None and channel < len(ds.channel_keys):
        ch_key = ds.channel_keys[channel]
    fig.suptitle(f"Sky components — channel {channel}" + (f" ({ch_key})" if ch_key else ""))
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    _finish(fig, savefig, show)
    return fig, axes


def plot_parity_grid(
    predicted,
    observed,
    channel_keys=None,
    log_scale=True,
    title=None,
    figsize=None,
    n_rows=None,
    n_cols=None,
    axes=None,
    savefig=None,
    show=True,
):
    """
    Plot predicted vs observed scatter per channel.

    Parameters
    ----------
    predicted : list of 1D arrays
        Model predictions per channel (flattened/masked pixels).
    observed : list of 1D arrays
        Observed data per channel (same shape as predicted).
    channel_keys : list of str, optional
        Labels for each channel.
    log_scale : bool
        Use log-log axes.
    savefig : str, optional
        Path to save the figure.

    Returns
    -------
    fig, axes
    """
    n_channels = len(predicted)
    fig, axes = _setup_grid(n_channels, figsize=figsize, n_rows=n_rows or 1, n_cols=n_cols, axes=axes)

    for idx in range(n_channels):
        ax = axes[idx]
        obs = np.asarray(observed[idx]).ravel()
        pred = np.asarray(predicted[idx]).ravel()

        if len(obs) > 0:
            ax.scatter(obs, pred, alpha=0.3, s=1, color="steelblue")
            lo, hi = np.nanmin(obs[obs > 0]) if np.any(obs > 0) else 1e-3, np.nanmax(obs)
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")

        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        label = channel_keys[idx] if channel_keys and idx < len(channel_keys) else f"Channel {idx}"
        ax.set_title(label)
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    _finish(fig, savefig, show)
    return fig, axes


def plot_residual_histograms(
    residuals,
    channel_keys=None,
    bins=50,
    title=None,
    figsize=None,
    n_rows=None,
    n_cols=None,
    axes=None,
    savefig=None,
    show=True,
):
    """
    Plot residual histograms per channel with N(0,1) overlay.

    Parameters
    ----------
    residuals : list of 1D arrays
        Residuals per channel (e.g. from noise_weighted_residuals).
    channel_keys : list of str, optional
        Labels for each channel.
    bins : int
        Number of histogram bins.
    savefig : str, optional
        Path to save the figure.

    Returns
    -------
    fig, axes
    """
    n_channels = len(residuals)
    fig, axes = _setup_grid(n_channels, figsize=figsize, n_rows=n_rows or 1, n_cols=n_cols, axes=axes)

    for idx in range(n_channels):
        ax = axes[idx]
        res = np.asarray(residuals[idx]).ravel()

        if len(res) > 0:
            ax.hist(res, bins=bins, density=True, alpha=0.15, color="steelblue")
            ax.hist(res, bins=bins, density=True, histtype="step", color="steelblue", linewidth=1.5)

        # N(0,1) reference
        x = np.linspace(-4, 4, 200)
        ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "k--", alpha=0.5, label="N(0,1)")

        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
        label = channel_keys[idx] if channel_keys and idx < len(channel_keys) else f"Channel {idx}"
        ax.set_title(label)
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    _finish(fig, savefig, show)
    return fig, axes


def plot_power_spectrum(
    spectra,
    channel_keys=None,
    label=None,
    color=None,
    linestyle=None,
    k_unit="1/pixel",
    title=None,
    figsize=None,
    n_rows=None,
    n_cols=None,
    axes=None,
    savefig=None,
    show=True,
):
    """
    Plot power spectra in a channel grid. Call multiple times with same
    axes to overlay (e.g. data vs posterior).

    Parameters
    ----------
    spectra : list of (k_centers, power) tuples
        Output from diagnostics.metrics.power_spectrum(). One per channel.
    channel_keys : list of str, optional
        Subplot titles.
    label : str, optional
        Legend label for this set (applied to all channels).
    color : str, optional
        Line color for all channels in this call.
    linestyle : str, optional
        Line style for all channels in this call.
    k_unit : str
        Label for k-axis.
    axes : array of Axes, optional
        Existing axes to overlay on.
    savefig : str, optional
        Path to save the figure.

    Returns
    -------
    fig, axes
    """
    n_channels = len(spectra)
    fig, axes = _setup_grid(n_channels, figsize=figsize, n_rows=n_rows or 1, n_cols=n_cols, axes=axes)

    kwargs = dict(alpha=0.8)
    if color is not None:
        kwargs["color"] = color
    if linestyle is not None:
        kwargs["linestyle"] = linestyle

    for idx, (k_centers, pwr) in enumerate(spectra):
        ax = axes[idx]
        ax.loglog(k_centers, pwr, label=label if idx == 0 else None, **kwargs)
        if channel_keys and idx < len(channel_keys):
            ax.set_title(channel_keys[idx])
        ax.set_xlabel(f"k [{k_unit}]")
        ax.set_ylabel("Power")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    _finish(fig, savefig, show)
    return fig, axes


def _setup_grid(n_panels, wcs=None, figsize=None, n_rows=None, n_cols=None, axes=None):
    """Set up a grid of subplots."""
    if axes is not None:
        if hasattr(axes, "flatten"):
            axes = axes.flatten()
        elif not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        return axes[0].figure, axes

    if n_rows is not None and n_cols is not None:
        pass
    elif n_rows is not None:
        n_cols = int(np.ceil(n_panels / n_rows))
    elif n_cols is not None:
        n_rows = int(np.ceil(n_panels / n_cols))
    else:
        n_cols = int(np.ceil(np.sqrt(n_panels)))
        n_rows = int(np.ceil(n_panels / n_cols))

    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    subplot_kw = {"projection": wcs} if wcs is not None else {}
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, subplot_kw=subplot_kw)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    # Hide unused subplots
    for idx in range(n_panels, len(axes)):
        axes[idx].axis("off")

    return fig, axes


def _finish(fig, savefig, show):
    """Save and/or show the figure."""
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
