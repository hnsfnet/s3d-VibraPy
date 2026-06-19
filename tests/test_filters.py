import numpy as np
import pytest

import signalproc as sp
from tests.conftest import get_peak_magnitude, count_zero_crossings


class TestFIRLowpass:
    def test_filter_coefficients_shape(self, lowpass_filter_100hz):
        b, cutoff, order = lowpass_filter_100hz
        assert len(b) == order + 1

    def test_filter_coefficients_sum_to_one(self, lowpass_filter_100hz):
        b, _, _ = lowpass_filter_100hz
        assert np.sum(b) == pytest.approx(1.0, rel=0.001)

    def test_filter_coefficients_are_symmetric(self, lowpass_filter_100hz):
        b, _, _ = lowpass_filter_100hz
        assert np.allclose(b, b[::-1], atol=1e-10)

    def test_dc_gain_is_one(self, lowpass_filter_100hz):
        b, _, _ = lowpass_filter_100hz
        dc_gain = np.abs(np.fft.rfft(b)[0])
        assert dc_gain == pytest.approx(1.0, rel=0.01)

    def test_filter_output_length_same(self, noisy_sine_50hz, lowpass_filter_100hz):
        _, signal, _, _, _ = noisy_sine_50hz
        b, _, _ = lowpass_filter_100hz
        filtered = sp.apply_filter(signal, b, zero_phase=False)
        assert len(filtered) == len(signal)

    def test_high_frequency_attenuation(self, fs, lowpass_filter_100hz):
        b, cutoff, order = lowpass_filter_100hz
        duration = 1.0

        t, s_low = sp.generate_sine(fs=fs, duration=duration, frequency=50, amplitude=1.0)
        _, s_high = sp.generate_sine(fs=fs, duration=duration, frequency=200, amplitude=1.0)
        mixed = s_low + s_high

        filtered = sp.apply_filter(mixed, b, zero_phase=True)
        freqs_filt, mag_filt = sp.compute_fft(filtered, fs, window="hann", nfft=4096)

        peak_freq_filt, peak_mag_filt, _ = get_peak_magnitude(freqs_filt, mag_filt)
        assert peak_freq_filt == pytest.approx(50, abs=1.0)

        high_freq_idx = np.argmin(np.abs(freqs_filt - 200))
        high_freq_mag = mag_filt[high_freq_idx]
        assert high_freq_mag < 0.1

    def test_low_frequency_preserved(self, fs, lowpass_filter_100hz):
        b, cutoff, order = lowpass_filter_100hz
        duration = 1.0

        t, s = sp.generate_sine(fs=fs, duration=duration, frequency=30, amplitude=2.0)
        filtered = sp.apply_filter(s, b, zero_phase=True)

        assert np.max(filtered) == pytest.approx(2.0, rel=0.05)
        assert np.min(filtered) == pytest.approx(-2.0, rel=0.05)

    def test_higher_order_has_sharper_cutoff(self, fs):
        duration = 1.0
        _, s_high = sp.generate_sine(fs=fs, duration=duration, frequency=150, amplitude=1.0)

        b_low_order = sp.fir_lowpass(cutoff=100, fs=fs, order=15)
        b_high_order = sp.fir_lowpass(cutoff=100, fs=fs, order=63)

        filtered_low = sp.apply_filter(s_high, b_low_order, zero_phase=True)
        filtered_high = sp.apply_filter(s_high, b_high_order, zero_phase=True)

        rms_low = np.sqrt(np.mean(filtered_low ** 2))
        rms_high = np.sqrt(np.mean(filtered_high ** 2))

        assert rms_high < rms_low * 0.5

    def test_cutoff_frequency_response(self, fs, lowpass_filter_100hz):
        b, cutoff, _ = lowpass_filter_100hz
        w, h = np.fft.rfftfreq(len(b) * 16, d=1.0 / fs), np.abs(np.fft.rfft(b, n=len(b) * 16))

        cutoff_idx = np.argmin(np.abs(w - cutoff))
        gain_at_cutoff = h[cutoff_idx]

        assert gain_at_cutoff == pytest.approx(np.sqrt(0.5), rel=0.2)


class TestZeroPhaseFiltering:
    def test_zero_phase_preserves_alignment(self, fs, lowpass_filter_100hz):
        b, cutoff, order = lowpass_filter_100hz
        duration = 0.1

        t, clean = sp.generate_sine(fs=fs, duration=duration, frequency=50, amplitude=2.0)
        _, noise = sp.generate_white_noise(fs=fs, duration=duration, amplitude=0.3, seed=42)
        mixed = clean + noise

        filtered_normal = sp.apply_filter(mixed, b, zero_phase=False)
        filtered_zero = sp.apply_filter(mixed, b, zero_phase=True)

        search_range = int(0.02 * fs)
        peak_clean = np.argmax(clean[:search_range])
        peak_normal = np.argmax(filtered_normal[:search_range])
        peak_zero = np.argmax(filtered_zero[:search_range])

        assert abs(peak_zero - peak_clean) <= 1
        assert abs(peak_normal - peak_clean) >= order // 4

    def test_zero_phase_crossing_alignment(self, fs, lowpass_filter_100hz):
        b, cutoff, order = lowpass_filter_100hz
        duration = 0.1

        t, clean = sp.generate_sine(fs=fs, duration=duration, frequency=50, amplitude=2.0)
        _, noise = sp.generate_white_noise(fs=fs, duration=duration, amplitude=0.2, seed=123)
        mixed = clean + noise

        filtered_zero = sp.apply_filter(mixed, b, zero_phase=True)

        clean_crossings = np.where(np.diff(np.sign(clean)))[0]
        filtered_crossings = np.where(np.diff(np.sign(filtered_zero)))[0]

        min_len = min(len(clean_crossings), len(filtered_crossings))
        avg_diff = np.mean(np.abs(clean_crossings[:min_len] - filtered_crossings[:min_len]))

        assert avg_diff < 2.0

    def test_zero_phase_symmetric_impulse_response(self, lowpass_filter_100hz):
        b, _, _ = lowpass_filter_100hz
        impulse = np.zeros(1000)
        impulse[500] = 1.0

        response = sp.apply_filter(impulse, b, zero_phase=True)

        left_half = response[400:500]
        right_half = response[501:601][::-1]

        assert np.allclose(left_half, right_half, atol=1e-10)

    def test_filtered_signal_can_be_used_for_statistics(self, noisy_sine_50hz, lowpass_filter_100hz):
        _, signal, _, _, _ = noisy_sine_50hz
        b, _, _ = lowpass_filter_100hz

        filtered = sp.apply_filter(signal, b, zero_phase=True)
        stats = sp.compute_statistics(filtered)

        assert "mean" in stats
        assert "variance" in stats
        assert "peak" in stats
        assert "peak_to_peak" in stats
        assert "rms" in stats

        assert isinstance(stats["rms"], float)
        assert stats["rms"] > 0
