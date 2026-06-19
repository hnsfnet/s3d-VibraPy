from signalproc import (
    generate_sine,
    generate_square,
    generate_triangle,
    generate_white_noise,
    compute_statistics,
    compute_fft,
    compute_psd_welch,
    fir_lowpass,
    apply_filter,
)

from visualizer import (
    plot_waveform,
    plot_spectrum,
    plot_psd,
    plot_filter_comparison,
)

from typing import Optional, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_time_domain(*args, **kwargs):
    return plot_waveform(*args, **kwargs)


def plot_fft_spectrum(*args, **kwargs):
    return plot_spectrum(*args, **kwargs)


def plot_filter_compare_time(
    t: np.ndarray,
    signal_before: np.ndarray,
    signal_after: np.ndarray,
    params: Optional[Dict[str, float]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(t, signal_before, linewidth=0.8, color="#1f77b4", label="Before filter")
    ax1.set_ylabel("Amplitude", fontsize=11)
    ax1.grid(True, which="both", linestyle="--", alpha=0.7)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_title("Time Domain: Before vs After Filtering", fontsize=13, fontweight="bold")

    ax2.plot(t, signal_after, linewidth=0.8, color="#d62728", label="After filter")
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Amplitude", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.7)
    ax2.legend(loc="upper right", fontsize=10)

    if params:
        param_strs = []
        for key, value in params.items():
            if isinstance(value, float):
                param_strs.append(f"{key}={value:.2f}")
            else:
                param_strs.append(f"{key}={value}")
        fig.suptitle(", ".join(param_strs), fontsize=11, y=0.98)

    if show:
        fig.tight_layout()
        plt.show()

    return fig


def plot_filter_compare_freq(
    freqs_before: np.ndarray,
    mag_before: np.ndarray,
    freqs_after: np.ndarray,
    mag_after: np.ndarray,
    params: Optional[Dict[str, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(freqs_before, mag_before, linewidth=1.0, color="#ff7f0e", label="Before filter")
    ax1.set_ylabel("Magnitude", fontsize=11)
    ax1.grid(True, which="both", linestyle="--", alpha=0.7)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_title("Frequency Domain: Before vs After Filtering", fontsize=13, fontweight="bold")

    if xlim is not None:
        ax1.set_xlim(xlim)

    ax2.plot(freqs_after, mag_after, linewidth=1.0, color="#9467bd", label="After filter")
    ax2.set_xlabel("Frequency (Hz)", fontsize=11)
    ax2.set_ylabel("Magnitude", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.7)
    ax2.legend(loc="upper right", fontsize=10)

    if xlim is not None:
        ax2.set_xlim(xlim)

    if params:
        param_strs = []
        for key, value in params.items():
            if isinstance(value, float):
                param_strs.append(f"{key}={value:.2f}")
            else:
                param_strs.append(f"{key}={value}")
        fig.suptitle(", ".join(param_strs), fontsize=11, y=0.98)

    if show:
        fig.tight_layout()
        plt.show()

    return fig


__all__ = [
    "generate_sine",
    "generate_square",
    "generate_triangle",
    "generate_white_noise",
    "compute_statistics",
    "compute_fft",
    "compute_psd_welch",
    "fir_lowpass",
    "apply_filter",
    "plot_waveform",
    "plot_spectrum",
    "plot_psd",
    "plot_filter_comparison",
    "plot_time_domain",
    "plot_fft_spectrum",
    "plot_filter_compare_time",
    "plot_filter_compare_freq",
]
