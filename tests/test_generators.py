import numpy as np
import pytest

from tests.conftest import count_zero_crossings


class TestSineWave:
    def test_output_shape(self, pure_sine_50hz, sample_params):
        t, signal, freq, amp = pure_sine_50hz
        assert len(t) == sample_params["n_samples"]
        assert len(signal) == sample_params["n_samples"]
        assert t[0] == pytest.approx(0.0)
        assert t[-1] == pytest.approx(sample_params["duration"] - 1.0 / sample_params["fs"])

    def test_zero_crossings_match_frequency(self, pure_sine_50hz):
        t, signal, freq, amp = pure_sine_50hz
        n_cycles = freq * (t[-1] - t[0])
        expected_crossings = int(2 * n_cycles)
        actual_crossings = count_zero_crossings(signal)
        assert actual_crossings == pytest.approx(expected_crossings, abs=2)

    def test_amplitude_range(self, pure_sine_50hz):
        t, signal, freq, amp = pure_sine_50hz
        assert np.max(signal) == pytest.approx(amp, rel=0.01)
        assert np.min(signal) == pytest.approx(-amp, rel=0.01)

    def test_dc_offset(self, fs, duration):
        import signalproc as sp
        offset = 1.5
        _, signal = sp.generate_sine(fs=fs, duration=duration, frequency=10, amplitude=1, dc_offset=offset)
        assert np.mean(signal) == pytest.approx(offset, abs=0.01)

    def test_phase_shift(self, fs, duration):
        import signalproc as sp
        t1, s1 = sp.generate_sine(fs=fs, duration=duration, frequency=10, amplitude=1, phase=0)
        t2, s2 = sp.generate_sine(fs=fs, duration=duration, frequency=10, amplitude=1, phase=np.pi / 2)
        assert np.allclose(s1, s2, atol=0.1) is False
        assert s1[0] == pytest.approx(0.0, abs=0.01)
        assert s2[0] == pytest.approx(1.0, abs=0.01)


class TestWhiteNoise:
    def test_output_shape(self, white_noise, sample_params):
        t, signal, amp = white_noise
        assert len(t) == sample_params["n_samples"]
        assert len(signal) == sample_params["n_samples"]

    def test_mean_near_zero(self, white_noise):
        _, signal, amp = white_noise
        assert np.mean(signal) == pytest.approx(0.0, abs=0.05)

    def test_variance_matches_amplitude(self, white_noise):
        _, signal, amp = white_noise
        expected_var = amp ** 2
        assert np.var(signal) == pytest.approx(expected_var, rel=0.1)

    def test_different_seeds_produce_different_signals(self, fs, duration):
        import signalproc as sp
        _, s1 = sp.generate_white_noise(fs=fs, duration=duration, amplitude=1, seed=1)
        _, s2 = sp.generate_white_noise(fs=fs, duration=duration, amplitude=1, seed=2)
        _, s3 = sp.generate_white_noise(fs=fs, duration=duration, amplitude=1, seed=1)
        assert not np.allclose(s1, s2)
        assert np.allclose(s1, s3)

    def test_dc_offset(self, fs, duration):
        import signalproc as sp
        offset = 2.0
        _, signal = sp.generate_white_noise(fs=fs, duration=duration, amplitude=1, dc_offset=offset, seed=42)
        assert np.mean(signal) == pytest.approx(offset, abs=0.05)


class TestSquareWave:
    def test_output_shape(self, square_wave_10hz, sample_params):
        t, signal, freq, amp = square_wave_10hz
        assert len(t) == sample_params["n_samples"]
        assert len(signal) == sample_params["n_samples"]

    def test_amplitude_levels(self, square_wave_10hz):
        _, signal, freq, amp = square_wave_10hz
        assert np.max(signal) == pytest.approx(amp)
        assert np.min(signal) == pytest.approx(-amp)

    def test_values_only_two_levels(self, square_wave_10hz):
        _, signal, freq, amp = square_wave_10hz
        unique_vals = np.unique(np.round(signal, decimals=6))
        assert len(unique_vals) == 2
        assert set(unique_vals) == {pytest.approx(amp), pytest.approx(-amp)}

    def test_fundamental_frequency_magnitude(self, square_wave_10hz, fs):
        import signalproc as sp
        from tests.conftest import get_peak_magnitude

        _, signal, freq, amp = square_wave_10hz
        freqs, mag = sp.compute_fft(signal, fs, window="none", nfft=4096)
        peak_freq, peak_mag, _ = get_peak_magnitude(freqs, mag)

        expected_peak_freq = freq
        expected_peak_mag = 4 * amp / np.pi

        assert peak_freq == pytest.approx(expected_peak_freq, abs=0.5)
        assert peak_mag == pytest.approx(expected_peak_mag, rel=0.05)

    def test_duty_cycle(self, fs, duration):
        import signalproc as sp
        _, signal = sp.generate_square(fs=fs, duration=duration, frequency=5, amplitude=1, duty_cycle=0.25)
        high_ratio = np.sum(signal > 0) / len(signal)
        assert high_ratio == pytest.approx(0.25, rel=0.05)

    def test_dc_offset(self, fs, duration):
        import signalproc as sp
        offset = 1.0
        _, signal = sp.generate_square(fs=fs, duration=duration, frequency=10, amplitude=1, dc_offset=offset)
        assert np.max(signal) == pytest.approx(offset + 1)
        assert np.min(signal) == pytest.approx(offset - 1)


class TestTriangleWave:
    def test_output_shape(self, fs, duration, sample_params):
        import signalproc as sp
        t, signal = sp.generate_triangle(fs=fs, duration=duration, frequency=10, amplitude=1)
        assert len(t) == sample_params["n_samples"]
        assert len(signal) == sample_params["n_samples"]

    def test_amplitude_range(self, fs, duration):
        import signalproc as sp
        amp = 2.0
        _, signal = sp.generate_triangle(fs=fs, duration=duration, frequency=10, amplitude=amp)
        assert np.max(signal) == pytest.approx(amp, rel=0.01)
        assert np.min(signal) == pytest.approx(-amp, rel=0.01)

    def test_zero_crossings(self, fs, duration):
        import signalproc as sp
        freq = 20.0
        t, signal = sp.generate_triangle(fs=fs, duration=duration, frequency=freq, amplitude=1)
        n_cycles = freq * duration
        expected_crossings = int(2 * n_cycles)
        actual_crossings = count_zero_crossings(signal)
        assert actual_crossings == pytest.approx(expected_crossings, abs=2)
