import numpy as np
from typing import Tuple, Dict, Optional


# ==========================================================================
# generators - 信号生成
# ==========================================================================

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


# ==========================================================================
# statistics - 时域统计
# ==========================================================================

def compute_statistics(signal: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(signal)),
        "variance": float(np.var(signal)),
        "peak": float(np.max(signal)),
        "peak_to_peak": float(np.max(signal) - np.min(signal)),
        "rms": float(np.sqrt(np.mean(signal ** 2))),
    }


# ==========================================================================
# transforms - 频域变换
# ==========================================================================

def _window(n: int, window: str) -> np.ndarray:
    if window == "hann":
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))
    elif window == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
    elif window == "none" or window is None:
        return np.ones(n)
    else:
        raise ValueError(f"Unknown window type: {window}")


def compute_fft(
    signal: np.ndarray,
    fs: float,
    window: str = "hann",
    nfft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    if nfft is None:
        nfft = n

    win = _window(n, window)
    win_gain = np.mean(win)
    windowed = signal * win

    fft_vals = np.fft.rfft(windowed, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    magnitude = np.abs(fft_vals) / (n * win_gain)
    magnitude[1:] = magnitude[1:] * 2
    return freqs, magnitude


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

    window = _window(nperseg, "hamming")
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


# ==========================================================================
# filters - 数字滤波
# ==========================================================================

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

    window = _window(order + 1, "hamming")
    taps = taps * window
    taps = taps / np.sum(taps)

    return taps


def apply_filter(
    signal: np.ndarray,
    b: np.ndarray,
    zero_phase: bool = False,
) -> np.ndarray:
    if zero_phase:
        return _filtfilt(signal, b)
    else:
        return np.convolve(signal, b, mode="same")


def _filtfilt(signal: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(signal)
    order = len(b) - 1
    pad_len = order * 3

    if n <= pad_len:
        pad_len = n // 2

    pre_pad = 2.0 * signal[0] - signal[pad_len:0:-1] if pad_len > 0 else np.array([])
    if pad_len > 0:
        post_pad = 2.0 * signal[-1] - signal[-2:-pad_len-2:-1]
    else:
        post_pad = np.array([])

    padded = np.concatenate([pre_pad, signal, post_pad])

    y_forward = np.convolve(padded, b, mode="same")
    y_backward = np.convolve(y_forward[::-1], b, mode="same")
    y = y_backward[::-1]

    result = y[pad_len:pad_len + n]
    return result
