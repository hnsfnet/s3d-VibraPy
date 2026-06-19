import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import signalproc as sp


@pytest.fixture(scope="module")
def fs():
    return 1000.0


@pytest.fixture(scope="module")
def duration():
    return 2.0


@pytest.fixture(scope="module")
def sample_params(fs, duration):
    return {
        "fs": fs,
        "duration": duration,
        "n_samples": int(fs * duration),
    }


@pytest.fixture(scope="module")
def pure_sine_50hz(fs, duration):
    freq = 50.0
    amplitude = 2.0
    t, signal = sp.generate_sine(
        fs=fs,
        duration=duration,
        frequency=freq,
        amplitude=amplitude,
    )
    return t, signal, freq, amplitude


@pytest.fixture(scope="module")
def noisy_sine_50hz(fs, duration):
    freq = 50.0
    sine_amp = 2.0
    noise_amp = 0.5
    t, sine = sp.generate_sine(fs=fs, duration=duration, frequency=freq, amplitude=sine_amp)
    _, noise = sp.generate_white_noise(fs=fs, duration=duration, amplitude=noise_amp, seed=42)
    return t, sine + noise, freq, sine_amp, noise_amp


@pytest.fixture(scope="module")
def square_wave_10hz(fs, duration):
    freq = 10.0
    amplitude = 1.0
    t, signal = sp.generate_square(
        fs=fs,
        duration=duration,
        frequency=freq,
        amplitude=amplitude,
    )
    return t, signal, freq, amplitude


@pytest.fixture(scope="module")
def white_noise(fs, duration):
    amplitude = 1.0
    t, signal = sp.generate_white_noise(
        fs=fs,
        duration=duration,
        amplitude=amplitude,
        seed=12345,
    )
    return t, signal, amplitude


@pytest.fixture(scope="module")
def lowpass_filter_100hz(fs):
    cutoff = 100.0
    order = 63
    b = sp.fir_lowpass(cutoff=cutoff, fs=fs, order=order)
    return b, cutoff, order


def count_zero_crossings(signal: np.ndarray) -> int:
    return int(np.sum(np.diff(np.sign(signal)) != 0))


def get_peak_magnitude(freqs: np.ndarray, magnitude: np.ndarray, exclude_dc: bool = True) -> tuple:
    if exclude_dc:
        idx = np.argmax(magnitude[1:]) + 1
    else:
        idx = np.argmax(magnitude)
    return freqs[idx], magnitude[idx], idx
