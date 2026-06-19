import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

import signalproc as sp


_DEFAULT_COLORS = {
    "waveform": "#1f77b4",
    "spectrum": "#ff7f0e",
    "psd": "#2ca02c",
    "before": "#1f77b4",
    "after": "#d62728",
    "spectrum_after": "#9467bd",
}

_DEFAULT_STYLE = {
    "grid_linestyle": "--",
    "grid_alpha": 0.7,
    "title_fontsize": 14,
    "title_fontweight": "bold",
    "label_fontsize": 12,
    "tick_fontsize": 10,
    "linewidth": 1.0,
}


def _format_params(params: Optional[Dict[str, float]]) -> str:
    if not params:
        return ""
    parts = []
    for key, value in params.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _apply_axes_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=_DEFAULT_STYLE["title_fontsize"],
                 fontweight=_DEFAULT_STYLE["title_fontweight"])
    ax.set_xlabel(xlabel, fontsize=_DEFAULT_STYLE["label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=_DEFAULT_STYLE["label_fontsize"])
    ax.grid(True, which="both",
            linestyle=_DEFAULT_STYLE["grid_linestyle"],
            alpha=_DEFAULT_STYLE["grid_alpha"])
    ax.tick_params(axis="both", labelsize=_DEFAULT_STYLE["tick_fontsize"])


def plot_waveform(
    t: np.ndarray,
    signal: np.ndarray,
    signal_type: str = "Signal",
    params: Optional[Dict[str, float]] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = None

    ax.plot(t, signal, linewidth=_DEFAULT_STYLE["linewidth"],
            color=_DEFAULT_COLORS["waveform"])

    title = signal_type
    param_str = _format_params(params)
    if param_str:
        title = f"{signal_type} | {param_str}"

    _apply_axes_style(ax, title, "Time (s)", "Amplitude")

    if show and fig is not None:
        fig.tight_layout()
        plt.show()

    return fig


def plot_spectrum(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    signal_type: str = "Signal",
    params: Optional[Dict[str, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = None

    ax.plot(freqs, magnitude, linewidth=_DEFAULT_STYLE["linewidth"] + 0.2,
            color=_DEFAULT_COLORS["spectrum"])

    if xlim is not None:
        ax.set_xlim(xlim)

    title = f"{signal_type} - FFT Spectrum"
    param_str = _format_params(params)
    if param_str:
        title = f"{title} | {param_str}"

    _apply_axes_style(ax, title, "Frequency (Hz)", "Magnitude")

    if show and fig is not None:
        fig.tight_layout()
        plt.show()

    return fig


def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    signal_type: str = "Signal",
    params: Optional[Dict[str, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = None

    ax.semilogy(freqs, psd, linewidth=_DEFAULT_STYLE["linewidth"] + 0.2,
                color=_DEFAULT_COLORS["psd"])

    if xlim is not None:
        ax.set_xlim(xlim)

    title = f"{signal_type} - PSD (Welch)"
    param_str = _format_params(params)
    if param_str:
        title = f"{title} | {param_str}"

    _apply_axes_style(ax, title, "Frequency (Hz)", "PSD (V\u00b2/Hz)")

    if show and fig is not None:
        fig.tight_layout()
        plt.show()

    return fig


def plot_filter_comparison(
    t: np.ndarray,
    signal_before: np.ndarray,
    signal_after: np.ndarray,
    fs: float,
    params: Optional[Dict[str, float]] = None,
    xlim_freq: Optional[Tuple[float, float]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    freqs_before, mag_before = sp.compute_fft(signal_before, fs)
    freqs_after, mag_after = sp.compute_fft(signal_after, fs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(t, signal_before, linewidth=0.8,
                    color=_DEFAULT_COLORS["before"], label="Before")
    axes[0, 0].set_title("Time Domain - Before Filtering", fontsize=12,
                         fontweight="bold")
    axes[0, 0].set_ylabel("Amplitude", fontsize=11)
    axes[0, 0].grid(True, which="both", linestyle="--", alpha=0.7)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].tick_params(axis="both", labelsize=9)

    axes[1, 0].plot(t, signal_after, linewidth=0.8,
                    color=_DEFAULT_COLORS["after"], label="After")
    axes[1, 0].set_title("Time Domain - After Filtering", fontsize=12,
                         fontweight="bold")
    axes[1, 0].set_xlabel("Time (s)", fontsize=11)
    axes[1, 0].set_ylabel("Amplitude", fontsize=11)
    axes[1, 0].grid(True, which="both", linestyle="--", alpha=0.7)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].tick_params(axis="both", labelsize=9)

    axes[0, 1].plot(freqs_before, mag_before, linewidth=1.0,
                    color=_DEFAULT_COLORS["spectrum"], label="Before")
    axes[0, 1].set_title("Frequency Domain - Before Filtering", fontsize=12,
                         fontweight="bold")
    axes[0, 1].set_ylabel("Magnitude", fontsize=11)
    axes[0, 1].grid(True, which="both", linestyle="--", alpha=0.7)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].tick_params(axis="both", labelsize=9)
    if xlim_freq is not None:
        axes[0, 1].set_xlim(xlim_freq)

    axes[1, 1].plot(freqs_after, mag_after, linewidth=1.0,
                    color=_DEFAULT_COLORS["spectrum_after"], label="After")
    axes[1, 1].set_title("Frequency Domain - After Filtering", fontsize=12,
                         fontweight="bold")
    axes[1, 1].set_xlabel("Frequency (Hz)", fontsize=11)
    axes[1, 1].set_ylabel("Magnitude", fontsize=11)
    axes[1, 1].grid(True, which="both", linestyle="--", alpha=0.7)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].tick_params(axis="both", labelsize=9)
    if xlim_freq is not None:
        axes[1, 1].set_xlim(xlim_freq)

    param_str = _format_params(params)
    if param_str:
        fig.suptitle(param_str, fontsize=12, y=0.995)

    if show:
        fig.tight_layout()
        plt.show()

    return fig
