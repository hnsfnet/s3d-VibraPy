import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


def _time_axis(fs: float, duration: float) -> np.ndarray:
    n_samples = int(fs * duration)
    return np.linspace(0, duration, n_samples, endpoint=False)


def generate_sine(
    fs: float,
    duration: float,
    frequency: float,
    amplitude: float = 1.0,
    dc_offset: float = 0.0,
    phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase) + dc_offset
    return t, signal


def generate_square(
    fs: float,
    duration: float,
    frequency: float,
    amplitude: float = 1.0,
    dc_offset: float = 0.0,
    duty_cycle: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    period = 1.0 / frequency
    phase = (t % period) / period
    wave = np.where(phase < duty_cycle, 1.0, -1.0)
    signal = amplitude * wave + dc_offset
    return t, signal


def generate_triangle(
    fs: float,
    duration: float,
    frequency: float,
    amplitude: float = 1.0,
    dc_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    period = 1.0 / frequency
    phase = (t % period) / period
    wave = 4.0 * np.abs(phase - 0.5) - 1.0
    signal = amplitude * wave + dc_offset
    return t, signal


def generate_white_noise(
    fs: float,
    duration: float,
    amplitude: float = 1.0,
    dc_offset: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    rng = np.random.default_rng(seed)
    signal = amplitude * rng.standard_normal(len(t)) + dc_offset
    return t, signal


def compute_statistics(signal: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(signal)),
        "variance": float(np.var(signal)),
        "peak": float(np.max(signal)),
        "peak_to_peak": float(np.max(signal) - np.min(signal)),
        "rms": float(np.sqrt(np.mean(signal ** 2))),
    }


def plot_time_domain(
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

    ax.plot(t, signal, linewidth=1.0, color="#1f77b4")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)

    title_parts = [signal_type]
    if params:
        param_strs = []
        for key, value in params.items():
            if isinstance(value, float):
                param_strs.append(f"{key}={value:.2f}")
            else:
                param_strs.append(f"{key}={value}")
        title_parts.append(", ".join(param_strs))
    ax.set_title(" | ".join(title_parts), fontsize=14, fontweight="bold")

    ax.tick_params(axis="both", labelsize=10)

    if show and fig is not None:
        fig.tight_layout()
        plt.show()

    return fig


def compute_fft(
    signal: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(signal)
    magnitude = np.abs(fft_vals) / n
    magnitude[1:] = magnitude[1:] * 2
    return freqs, magnitude


def plot_fft_spectrum(
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

    ax.plot(freqs, magnitude, linewidth=1.2, color="#ff7f0e")
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)

    title_parts = [f"{signal_type} - FFT Spectrum"]
    if params:
        param_strs = []
        for key, value in params.items():
            if isinstance(value, float):
                param_strs.append(f"{key}={value:.2f}")
            else:
                param_strs.append(f"{key}={value}")
        title_parts.append(", ".join(param_strs))
    ax.set_title(" | ".join(title_parts), fontsize=14, fontweight="bold")

    ax.tick_params(axis="both", labelsize=10)

    if show and fig is not None:
        fig.tight_layout()
        plt.show()

    return fig


def compute_psd_welch(
    signal: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    if nperseg is None:
        nperseg = min(256, n // 4)
    if noverlap is None:
        noverlap = nperseg // 2

    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1))
    win_power = np.sum(window ** 2)

    step = nperseg - noverlap
    n_segments = (n - nperseg) // step + 1

    psd_accum = np.zeros(nperseg // 2 + 1)

    for i in range(n_segments):
        start = i * step
        segment = signal[start:start + nperseg] * window
        fft_seg = np.fft.rfft(segment)
        psd_accum += np.abs(fft_seg) ** 2

    psd = psd_accum / (n_segments * win_power * fs)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)

    return freqs, psd


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

    ax.semilogy(freqs, psd, linewidth=1.2, color="#2ca02c")
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("PSD (V\u00b2/Hz)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)

    title_parts = [f"{signal_type} - PSD (Welch)"]
    if params:
        param_strs = []
        for key, value in params.items():
            if isinstance(value, float):
                param_strs.append(f"{key}={value:.2f}")
            else:
                param_strs.append(f"{key}={value}")
        title_parts.append(", ".join(param_strs))
    ax.set_title(" | ".join(title_parts), fontsize=14, fontweight="bold")

    ax.tick_params(axis="both", labelsize=10)

    if show and fig is not None:
        fig.tight_layout()
        plt.show()

    return fig


def fir_lowpass(
    cutoff: float,
    fs: float,
    order: int = 31,
) -> np.ndarray:
    nyquist = fs / 2.0
    wc = cutoff / nyquist

    taps = np.zeros(order + 1)
    mid = order // 2

    for n in range(order + 1):
        if n == mid:
            taps[n] = wc
        else:
            x = n - mid
            taps[n] = np.sin(wc * np.pi * x) / (np.pi * x)

    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(order + 1) / order)
    taps = taps * window
    taps = taps / np.sum(taps)

    return taps


def apply_filter(
    signal: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return np.convolve(signal, b, mode="same")


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
