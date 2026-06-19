import numpy as np
import pytest

import signalproc as sp
from tests.conftest import get_peak_magnitude


class TestFullProcessingPipeline:
    def test_complete_pipeline_rms_reduction(self, noisy_sine_50hz, fs, lowpass_filter_100hz):
        t, mixed, freq, sine_amp, noise_amp = noisy_sine_50hz
        b, cutoff, order = lowpass_filter_100hz

        stats_before = sp.compute_statistics(mixed)
        freqs_before, mag_before = sp.compute_fft(mixed, fs, window="hann", nfft=2048)

        filtered = sp.apply_filter(mixed, b, zero_phase=True)

        stats_after = sp.compute_statistics(filtered)
        freqs_after, mag_after = sp.compute_fft(filtered, fs, window="hann", nfft=2048)

        assert stats_after["rms"] < stats_before["rms"]
        assert stats_after["variance"] < stats_before["variance"]

        rms_reduction = (stats_before["rms"] - stats_after["rms"]) / stats_before["rms"]
        assert rms_reduction > 0.02

        peak_freq_before, peak_mag_before, _ = get_peak_magnitude(freqs_before, mag_before)
        peak_freq_after, peak_mag_after, _ = get_peak_magnitude(freqs_after, mag_after)

        assert peak_freq_before == pytest.approx(freq, abs=0.5)
        assert peak_freq_after == pytest.approx(freq, abs=0.5)

        peak_mag_ratio = peak_mag_after / peak_mag_before
        assert peak_mag_ratio == pytest.approx(1.0, rel=0.1)

    def test_peak_position_preserved_after_zero_phase_filtering(
        self, noisy_sine_50hz, fs, lowpass_filter_100hz
    ):
        t, mixed, freq, sine_amp, noise_amp = noisy_sine_50hz
        b, cutoff, order = lowpass_filter_100hz

        filtered = sp.apply_filter(mixed, b, zero_phase=True)

        search_end = int(0.1 * fs)
        peak_idx_raw = np.argmax(mixed[:search_end])
        peak_idx_filt = np.argmax(filtered[:search_end])

        assert abs(peak_idx_filt - peak_idx_raw) <= 2
        assert t[peak_idx_filt] == pytest.approx(t[peak_idx_raw], abs=0.002)

    def test_high_frequency_noise_attenuated_in_spectrum(
        self, noisy_sine_50hz, fs, lowpass_filter_100hz
    ):
        t, mixed, freq, sine_amp, noise_amp = noisy_sine_50hz
        b, cutoff, order = lowpass_filter_100hz

        freqs_before, mag_before = sp.compute_fft(mixed, fs, window="hann", nfft=2048)
        filtered = sp.apply_filter(mixed, b, zero_phase=True)
        freqs_after, mag_after = sp.compute_fft(filtered, fs, window="hann", nfft=2048)

        high_freq_mask = freqs_before > cutoff * 1.5
        noise_energy_before = np.sum(mag_before[high_freq_mask] ** 2)
        noise_energy_after = np.sum(mag_after[high_freq_mask] ** 2)

        assert noise_energy_after < noise_energy_before * 0.2

    def test_main_frequency_component_preserved(
        self, noisy_sine_50hz, fs, lowpass_filter_100hz
    ):
        t, mixed, freq, sine_amp, noise_amp = noisy_sine_50hz
        b, cutoff, order = lowpass_filter_100hz

        freqs_before, mag_before = sp.compute_fft(mixed, fs, window="hann", nfft=4096)
        filtered = sp.apply_filter(mixed, b, zero_phase=True)
        freqs_after, mag_after = sp.compute_fft(filtered, fs, window="hann", nfft=4096)

        _, peak_mag_before, peak_idx_before = get_peak_magnitude(freqs_before, mag_before)
        _, peak_mag_after, peak_idx_after = get_peak_magnitude(freqs_after, mag_after)

        assert freqs_before[peak_idx_before] == pytest.approx(freq, abs=0.5)
        assert freqs_after[peak_idx_after] == pytest.approx(freq, abs=0.5)

        assert peak_mag_after == pytest.approx(peak_mag_before, rel=0.15)

    def test_statistics_consistency_after_multiple_filterings(
        self, noisy_sine_50hz, fs, lowpass_filter_100hz
    ):
        t, mixed, freq, sine_amp, noise_amp = noisy_sine_50hz
        b, cutoff, order = lowpass_filter_100hz

        filtered1 = sp.apply_filter(mixed, b, zero_phase=True)
        filtered2 = sp.apply_filter(mixed, b, zero_phase=True)

        assert np.allclose(filtered1, filtered2)

        stats1 = sp.compute_statistics(filtered1)
        stats2 = sp.compute_statistics(filtered2)

        for key in stats1:
            assert stats1[key] == pytest.approx(stats2[key])

    def test_psd_shows_noise_reduction(
        self, noisy_sine_50hz, fs, lowpass_filter_100hz
    ):
        t, mixed, freq, sine_amp, noise_amp = noisy_sine_50hz
        b, cutoff, order = lowpass_filter_100hz

        freqs_psd_raw, psd_raw = sp.compute_psd_welch(mixed, fs, nperseg=256)
        filtered = sp.apply_filter(mixed, b, zero_phase=True)
        freqs_psd_filt, psd_filt = sp.compute_psd_welch(filtered, fs, nperseg=256)

        high_freq_mask = freqs_psd_raw > cutoff * 1.5
        avg_psd_raw_high = np.mean(psd_raw[high_freq_mask])
        avg_psd_filt_high = np.mean(psd_filt[high_freq_mask])

        assert avg_psd_filt_high < avg_psd_raw_high * 0.1

        signal_band_mask = (freqs_psd_raw > freq - 5) & (freqs_psd_raw < freq + 5)
        avg_psd_raw_signal = np.mean(psd_raw[signal_band_mask])
        avg_psd_filt_signal = np.mean(psd_filt[signal_band_mask])

        assert avg_psd_filt_signal == pytest.approx(avg_psd_raw_signal, rel=0.3)


class TestSpectrumComparisonBeforeAfterFilter:
    def test_50hz_peak_amplitude_preserved(
        self, noisy_sine_50hz, fs, lowpass_filter_100hz
    ):
        t, mixed, freq, sine_amp, noise_amp = noisy_sine_50hz
        b, cutoff, order = lowpass_filter_100hz

        freqs_before, mag_before = sp.compute_fft(mixed, fs, window="hann", nfft=4096)
        filtered = sp.apply_filter(mixed, b, zero_phase=True)
        freqs_after, mag_after = sp.compute_fft(filtered, fs, window="hann", nfft=4096)

        target_idx_before = np.argmin(np.abs(freqs_before - freq))
        target_idx_after = np.argmin(np.abs(freqs_after - freq))

        mag_before_at_50hz = mag_before[target_idx_before]
        mag_after_at_50hz = mag_after[target_idx_after]

        assert mag_before_at_50hz == pytest.approx(sine_amp, rel=0.15)
        assert mag_after_at_50hz == pytest.approx(sine_amp, rel=0.15)
        assert mag_after_at_50hz == pytest.approx(mag_before_at_50hz, rel=0.1)

    def test_high_frequency_band_attenuated(
        self, noisy_sine_50hz, fs, lowpass_filter_100hz
    ):
        t, mixed, freq, sine_amp, noise_amp = noisy_sine_50hz
        b, cutoff, order = lowpass_filter_100hz

        freqs_before, mag_before = sp.compute_fft(mixed, fs, window="hann", nfft=2048)
        filtered = sp.apply_filter(mixed, b, zero_phase=True)
        freqs_after, mag_after = sp.compute_fft(filtered, fs, window="hann", nfft=2048)

        high_freq_start = cutoff * 2.0
        high_freq_mask = freqs_before >= high_freq_start

        avg_mag_before_high = np.mean(mag_before[high_freq_mask])
        avg_mag_after_high = np.mean(mag_after[high_freq_mask])

        assert avg_mag_after_high < avg_mag_before_high * 0.2
        assert avg_mag_after_high < 0.1 * noise_amp

    def test_signal_to_noise_ratio_improved(
        self, noisy_sine_50hz, fs, lowpass_filter_100hz
    ):
        t, mixed, freq, sine_amp, noise_amp = noisy_sine_50hz
        b, cutoff, order = lowpass_filter_100hz

        def compute_snr(freqs, mag, signal_freq, bandwidth=2.0):
            signal_mask = (freqs >= signal_freq - bandwidth) & (freqs <= signal_freq + bandwidth)
            noise_mask = ~signal_mask & (freqs >= 1.0)

            signal_power = np.sum(mag[signal_mask] ** 2)
            noise_power = np.sum(mag[noise_mask] ** 2)

            if noise_power == 0:
                return float("inf")
            return 10 * np.log10(signal_power / noise_power)

        freqs_before, mag_before = sp.compute_fft(mixed, fs, window="hann", nfft=4096)
        filtered = sp.apply_filter(mixed, b, zero_phase=True)
        freqs_after, mag_after = sp.compute_fft(filtered, fs, window="hann", nfft=4096)

        snr_before = compute_snr(freqs_before, mag_before, freq)
        snr_after = compute_snr(freqs_after, mag_after, freq)

        assert snr_after > snr_before
        assert snr_after - snr_before > 3.0

    def test_dc_component_unchanged(self, fs, lowpass_filter_100hz):
        b, cutoff, order = lowpass_filter_100hz
        duration = 1.0
        offset = 2.0

        t, sine = sp.generate_sine(fs=fs, duration=duration, frequency=50, amplitude=1, dc_offset=offset)
        _, noise = sp.generate_white_noise(fs=fs, duration=duration, amplitude=0.2, seed=42)
        mixed = sine + noise

        filtered = sp.apply_filter(mixed, b, zero_phase=True)

        assert np.mean(filtered) == pytest.approx(offset, abs=0.05)
        assert np.mean(mixed) == pytest.approx(offset, abs=0.05)
