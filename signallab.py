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
