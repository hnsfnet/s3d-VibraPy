import numpy as np
import matplotlib.pyplot as plt

import signalproc as sp
import visualizer as vz


def demo_signal_generation_and_stats():
    print("\n" + "=" * 70)
    print("[1] 信号生成 + 时域统计")
    print("=" * 70)

    fs = 1000.0
    duration = 1.0

    t, sine = sp.generate_sine(fs=fs, duration=duration, frequency=50.0, amplitude=2.0)
    _, noise = sp.generate_white_noise(fs=fs, duration=duration, amplitude=0.5, seed=42)
    mixed = sine + noise

    stats = sp.compute_statistics(mixed)
    print(f"  均值     : {stats['mean']:.6f}")
    print(f"  方差     : {stats['variance']:.6f}")
    print(f"  峰值     : {stats['peak']:.4f} V")
    print(f"  峰峰值   : {stats['peak_to_peak']:.4f} V")
    print(f"  RMS      : {stats['rms']:.4f} V")

    vz.plot_waveform(
        t=t,
        signal=mixed,
        signal_type="Raw Signal (Sine + Noise)",
        params={"fs": fs, "freq": 50.0, "sine_amp": 2.0, "noise_amp": 0.5},
        show=False,
    )

    return t, mixed, fs


def demo_freq_domain(t, signal, fs):
    print("\n" + "=" * 70)
    print("[2] 频域分析 (FFT + PSD)")
    print("=" * 70)

    freqs_fft, mag_fft = sp.compute_fft(signal, fs, window="hann", nfft=2048)
    peak_idx = np.argmax(mag_fft[1:]) + 1
    print(f"  FFT 峰值频率  : {freqs_fft[peak_idx]:.2f} Hz")
    print(f"  FFT 峰值幅度  : {mag_fft[peak_idx]:.4f} V")

    vz.plot_spectrum(
        freqs=freqs_fft,
        magnitude=mag_fft,
        signal_type="Raw Signal",
        params={"window": "hann", "nfft": 2048},
        xlim=(0, fs / 2),
        show=False,
    )

    freqs_psd, psd = sp.compute_psd_welch(signal, fs, nperseg=256)
    print(f"  PSD 点数     : {len(psd)}")

    vz.plot_psd(
        freqs=freqs_psd,
        psd=psd,
        signal_type="Raw Signal",
        params={"nperseg": 256, "noverlap": 128},
        xlim=(0, fs / 2),
        show=False,
    )


def demo_filtering(t, signal, fs):
    print("\n" + "=" * 70)
    print("[3] FIR 低通滤波 (零相位)")
    print("=" * 70)

    cutoff = 100.0
    order = 63

    b = sp.fir_lowpass(cutoff=cutoff, fs=fs, order=order)
    filtered = sp.apply_filter(signal, b, zero_phase=True)
    print(f"  截止频率: {cutoff} Hz")
    print(f"  滤波器阶数: {order}")
    print(f"  抽头数: {len(b)}")

    vz.plot_filter_comparison(
        t=t,
        signal_before=signal,
        signal_after=filtered,
        fs=fs,
        params={"cutoff": cutoff, "order": order, "zero_phase": True},
        xlim_freq=(0, fs / 2),
        show=False,
    )

    stats_before = sp.compute_statistics(signal)
    stats_after = sp.compute_statistics(filtered)
    print("\n  滤波前后统计对比:")
    print(f"  {'指标':<10} {'滤波前':>12} {'滤波后':>12} {'变化':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    for key in ["mean", "variance", "peak", "peak_to_peak", "rms"]:
        v_before = stats_before[key]
        v_after = stats_after[key]
        diff = v_after - v_before
        print(f"  {key:<10} {v_before:>12.6f} {v_after:>12.6f} {diff:>+10.6f}")

    return filtered


def demo_window_verification():
    print("\n" + "=" * 70)
    print("[验证] 加窗抑制频谱泄漏")
    print("=" * 70)

    fs = 1000.0
    duration = 1.0
    freq = 50.0
    amp = 2.0

    t, sine = sp.generate_sine(fs=fs, duration=duration, frequency=freq, amplitude=amp)

    freqs_nowin, mag_nowin = sp.compute_fft(sine, fs, window="none")
    freqs_hann, mag_hann = sp.compute_fft(sine, fs, window="hann", nfft=4096)

    peak_nowin = np.argmax(mag_nowin[1:]) + 1
    peak_hann = np.argmax(mag_hann[1:]) + 1

    print(f"  不加窗: 峰值 {freqs_nowin[peak_nowin]:.2f}Hz / {mag_nowin[peak_nowin]:.4f}V")
    print(f"  加 Hann 窗: 峰值 {freqs_hann[peak_hann]:.2f}Hz / {mag_hann[peak_hann]:.4f}V")
    print(f"  理论值: {freq}Hz / {amp}V")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    vz.plot_spectrum(freqs_nowin, mag_nowin, "No Window",
                     show=False, ax=ax1)
    vz.plot_spectrum(freqs_hann, mag_hann, "Hann Window",
                     show=False, ax=ax2)
    ax1.set_xlim(30, 70)
    ax2.set_xlim(30, 70)
    fig.suptitle("Spectrum Leakage: No Window vs Hann Window",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    return fig


def demo_zero_phase_verification():
    print("\n" + "=" * 70)
    print("[验证] 零相位滤波对齐效果")
    print("=" * 70)

    fs = 1000.0
    duration = 0.05
    freq = 50.0
    amp = 2.0

    t, clean = sp.generate_sine(fs=fs, duration=duration, frequency=freq, amplitude=amp)
    _, noise = sp.generate_white_noise(fs=fs, duration=duration, amplitude=0.3, seed=42)
    mixed = clean + noise

    b = sp.fir_lowpass(cutoff=100.0, fs=fs, order=31)
    y_normal = sp.apply_filter(mixed, b, zero_phase=False)
    y_zero = sp.apply_filter(mixed, b, zero_phase=True)

    peak_clean = np.argmax(clean[:int(0.02 * fs)])
    peak_normal = np.argmax(y_normal[:int(0.02 * fs)])
    peak_zero = np.argmax(y_zero[:int(0.02 * fs)])

    print(f"  干净信号峰值: {t[peak_clean]*1000:.2f} ms")
    print(f"  普通滤波峰值: {t[peak_normal]*1000:.2f} ms (偏移 {peak_normal - peak_clean} 点)")
    print(f"  零相位峰值  : {t[peak_zero]*1000:.2f} ms (偏移 {peak_zero - peak_clean} 点)")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t * 1000, mixed, color="#1f77b4", linewidth=0.8, label="Noisy")
    axes[0].plot(t * 1000, clean, color="black", linestyle="--", linewidth=1.5, label="Clean (ref)")
    axes[0].set_title("Raw (Noisy) + Clean Reference", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend(fontsize=9)

    axes[1].plot(t * 1000, y_normal, color="#d62728", linewidth=0.8, label="Normal FIR")
    axes[1].plot(t * 1000, clean, color="black", linestyle="--", linewidth=1.5, label="Clean (ref)")
    axes[1].set_title("Normal FIR (phase shift)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Amplitude", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend(fontsize=9)

    axes[2].plot(t * 1000, y_zero, color="#2ca02c", linewidth=0.8, label="Zero-phase")
    axes[2].plot(t * 1000, clean, color="black", linestyle="--", linewidth=1.5, label="Clean (ref)")
    axes[2].set_title("Zero-phase (filtfilt) - aligned!", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time (ms)", fontsize=10)
    axes[2].set_ylabel("Amplitude", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.7)
    axes[2].legend(fontsize=9)

    fig.suptitle("Phase Alignment: Normal FIR vs Zero-Phase",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    return fig


def main():
    print("=" * 70)
    print("SignalLab Demo v3 - 重构版 (signalproc + visualizer)")
    print("=" * 70)

    t, mixed, fs = demo_signal_generation_and_stats()
    demo_freq_domain(t, mixed, fs)
    demo_filtering(t, mixed, fs)

    fig_win = demo_window_verification()
    fig_phase = demo_zero_phase_verification()

    print("\n" + "=" * 70)
    print("所有图表已生成，请查看弹出窗口。")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
