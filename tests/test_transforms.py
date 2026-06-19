import numpy as np
import pytest

import signalproc as sp
from tests.conftest import get_peak_magnitude


class TestFFT:
    def test_peak_frequency_accuracy(self, pure_sine_50hz, fs):
        _, signal, freq, amp = pure_sine_50hz
        freqs, mag = sp.compute_fft(signal, fs, window="hann", nfft=4096)
        peak_freq, peak_mag, _ = get_peak_magnitude(freqs, mag)
        assert peak_freq == pytest.approx(freq, abs=0.5)

    def test_peak_magnitude_accuracy(self, pure_sine_50hz, fs):
        _, signal, freq, amp = pure_sine_50hz
        freqs, mag = sp.compute_fft(signal, fs, window="hann", nfft=4096)
        peak_freq, peak_mag, _ = get_peak_magnitude(freqs, mag)
        assert peak_mag == pytest.approx(amp, rel=0.1)

    def test_window_reduces_sidelobe_leakage(self, pure_sine_50hz, fs):
        _, signal, freq, amp = pure_sine_50hz

        freqs_nowin, mag_nowin = sp.compute_fft(signal, fs, window="none")
        freqs_hann, mag_hann = sp.compute_fft(signal, fs, window="hann")

        _, _, peak_idx_nowin = get_peak_magnitude(freqs_nowin, mag_nowin)
        _, _, peak_idx_hann = get_peak_magnitude(freqs_hann, mag_hann)

        sideband_range = 10
        start_nowin = max(1, peak_idx_nowin - sideband_range)
        end_nowin = peak_idx_nowin + sideband_range + 1
        sidelobe_nowin = np.sum(mag_nowin[start_nowin:peak_idx_nowin]) + \
                         np.sum(mag_nowin[peak_idx_nowin+1:end_nowin])

        start_hann = max(1, peak_idx_hann - sideband_range)
        end_hann = peak_idx_hann + sideband_range + 1
        sidelobe_hann = np.sum(mag_hann[start_hann:peak_idx_hann]) + \
                        np.sum(mag_hann[peak_idx_hann+1:end_hann])

        assert sidelobe_hann < sidelobe_nowin * 0.1

    def test_fft_output_length(self, pure_sine_50hz, fs):
        _, signal, freq, amp = pure_sine_50hz
        n = len(signal)

        freqs, mag = sp.compute_fft(signal, fs, window="hann")
        assert len(freqs) == n // 2 + 1
        assert len(mag) == n // 2 + 1

        nfft = 2048
        freqs2, mag2 = sp.compute_fft(signal, fs, window="hann", nfft=nfft)
        assert len(freqs2) == nfft // 2 + 1
        assert len(mag2) == nfft // 2 + 1

    def test_dc_component(self, fs, duration):
        offset = 3.0
        t, signal = sp.generate_sine(fs=fs, duration=duration, frequency=10, amplitude=1, dc_offset=offset)
        freqs, mag = sp.compute_fft(signal, fs, window="none")
        assert mag[0] == pytest.approx(offset, rel=0.05)

    def test_invalid_window_raises(self, pure_sine_50hz, fs):
        _, signal, _, _ = pure_sine_50hz
        with pytest.raises(ValueError, match="Unknown window type"):
            sp.compute_fft(signal, fs, window="invalid_window")

    def test_nyquist_frequency(self, pure_sine_50hz, fs):
        _, signal, _, _ = pure_sine_50hz
        freqs, mag = sp.compute_fft(signal, fs, window="hann")
        assert freqs[-1] == pytest.approx(fs / 2)
        assert freqs[0] == pytest.approx(0.0)

    def test_multiple_frequencies(self, fs, duration):
        t, s1 = sp.generate_sine(fs=fs, duration=duration, frequency=30, amplitude=1.0)
        _, s2 = sp.generate_sine(fs=fs, duration=duration, frequency=80, amplitude=0.5)
        mixed = s1 + s2

        freqs, mag = sp.compute_fft(mixed, fs, window="hann", nfft=4096)

        sorted_idx = np.argsort(mag[1:])[::-1] + 1
        peak1_freq = freqs[sorted_idx[0]]
        peak2_freq = freqs[sorted_idx[1]]

        assert peak1_freq == pytest.approx(30, abs=0.5) or peak1_freq == pytest.approx(80, abs=0.5)
        assert peak2_freq == pytest.approx(30, abs=0.5) or peak2_freq == pytest.approx(80, abs=0.5)


class TestPSDWelch:
    def test_output_shape(self, noisy_sine_50hz, fs):
        _, signal, _, _, _ = noisy_sine_50hz
        nperseg = 256
        freqs, psd = sp.compute_psd_welch(signal, fs, nperseg=nperseg)
        assert len(freqs) == nperseg // 2 + 1
        assert len(psd) == nperseg // 2 + 1

    def test_peak_frequency(self, noisy_sine_50hz, fs):
        _, signal, freq, _, _ = noisy_sine_50hz
        freqs, psd = sp.compute_psd_welch(signal, fs, nperseg=512)
        peak_idx = np.argmax(psd[1:]) + 1
        assert freqs[peak_idx] == pytest.approx(freq, abs=1.0)

    def test_noise_floor_reduced_after_filter(self, noisy_sine_50hz, fs, lowpass_filter_100hz):
        _, signal, _, _, _ = noisy_sine_50hz
        b, cutoff, order = lowpass_filter_100hz
        filtered = sp.apply_filter(signal, b, zero_phase=True)

        freqs_raw, psd_raw = sp.compute_psd_welch(signal, fs, nperseg=256)
        freqs_filt, psd_filt = sp.compute_psd_welch(filtered, fs, nperseg=256)

        high_freq_mask = freqs_raw > cutoff * 1.5
        noise_power_raw = np.mean(psd_raw[high_freq_mask])
        noise_power_filt = np.mean(psd_filt[high_freq_mask])

        assert noise_power_filt < noise_power_raw * 0.1

    def test_psd_values_positive(self, noisy_sine_50hz, fs):
        _, signal, _, _, _ = noisy_sine_50hz
        freqs, psd = sp.compute_psd_welch(signal, fs)
        assert np.all(psd >= 0)

    def test_default_nperseg(self, noisy_sine_50hz, fs):
        _, signal, _, _, _ = noisy_sine_50hz
        freqs, psd = sp.compute_psd_welch(signal, fs)
        expected_nperseg = min(256, len(signal) // 4)
        assert len(freqs) == expected_nperseg // 2 + 1
