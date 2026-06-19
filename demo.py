import signallab as sl
import numpy as np
import matplotlib.pyplot as plt


def demo_window_leak_fix():
    print("\n" + "=" * 70)
    print(" 验证 1: FFT 加窗前后对比 (修复频谱泄漏)")
    print("=" * 70)

    fs = 1000.0
    duration = 1.0
    sine_freq = 50.0
    sine_amp = 2.0

    t, sine = sl.generate_sine(fs=fs, duration=duration, frequency=sine_freq, amplitude=sine_amp)

    freqs_nowin, mag_nowin = sl.compute_fft(sine, fs, window="none")
    freqs_hann, mag_hann = sl.compute_fft(sine, fs, window="hann", nfft=4096)

    peak_idx_nowin = np.argmax(mag_nowin[1:]) + 1
    peak_idx_hann = np.argmax(mag_hann[1:]) + 1

    print(f"  纯正弦波: {sine_freq}Hz, 幅度 {sine_amp}V")
    print(f"  不加窗:  峰值频率 = {freqs_nowin[peak_idx_nowin]:.2f}Hz, 幅度 = {mag_nowin[peak_idx_nowin]:.4f}V")
    print(f"  加 Hann 窗: 峰值频率 = {freqs_hann[peak_idx_hann]:.2f}Hz, 幅度 = {mag_hann[peak_idx_hann]:.4f}V")
    print(f"  理论值:  频率 = {sine_freq}Hz, 幅度 = {sine_amp}V")

    sideband_range = 10
    leak_nowin = np.sum(mag_nowin[max(0, peak_idx_nowin - sideband_range):peak_idx_nowin])
    leak_hann = np.sum(mag_hann[max(0, peak_idx_hann - sideband_range):peak_idx_hann])
    print(f"  主峰旁 10Hz 内裙边能量(和): 不加窗={leak_nowin:.4f}, 加窗={leak_hann:.6f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(freqs_nowin, mag_nowin, linewidth=1.0, color="#d62728")
    ax1.set_title("FFT 不加窗 (有泄漏)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Magnitude", fontsize=11)
    ax1.grid(True, which="both", linestyle="--", alpha=0.7)
    ax1.set_xlim(30, 70)

    ax2.plot(freqs_hann, mag_hann, linewidth=1.0, color="#2ca02c")
    ax2.set_title("FFT + Hann 窗 (泄漏抑制)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frequency (Hz)", fontsize=11)
    ax2.set_ylabel("Magnitude", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.7)
    ax2.set_xlim(30, 70)

    fig.suptitle("Spectrum Leakage Comparison: 50Hz Sine Wave", fontsize=13, fontweight="bold")
    fig.tight_layout()

    return fig


def demo_zero_phase_fix():
    print("\n" + "=" * 70)
    print(" 验证 2: 零相位滤波对比 (修复相位偏移)")
    print("=" * 70)

    fs = 1000.0
    duration = 0.1
    sine_freq = 50.0
    sine_amp = 2.0
    noise_amp = 0.3

    fir_cutoff = 100.0
    fir_order = 63

    t, sine = sl.generate_sine(fs=fs, duration=duration, frequency=sine_freq, amplitude=sine_amp)
    _, noise = sl.generate_white_noise(fs=fs, duration=duration, amplitude=noise_amp, seed=42)
    mixed = sine + noise

    b = sl.fir_lowpass(cutoff=fir_cutoff, fs=fs, order=fir_order)

    filtered_normal = sl.apply_filter(mixed, b, zero_phase=False)
    filtered_zero = sl.apply_filter(mixed, b, zero_phase=True)

    group_delay = fir_order // 2
    print(f"  滤波器阶数: {fir_order}, 理论群延迟: {group_delay} 采样点 = {group_delay/fs*1000:.1f}ms")

    peak_idx_raw = np.argmax(mixed[:int(0.02 * fs)])
    peak_idx_normal = np.argmax(filtered_normal[:int(0.02 * fs)])
    peak_idx_zero = np.argmax(filtered_zero[:int(0.02 * fs)])

    print(f"  原始信号峰值位置: t = {t[peak_idx_raw]*1000:.2f} ms")
    print(f"  普通滤波峰值位置: t = {t[peak_idx_normal]*1000:.2f} ms (偏移 {(peak_idx_normal - peak_idx_raw)} 采样点)")
    print(f"  零相位滤波峰值位置: t = {t[peak_idx_zero]*1000:.2f} ms (偏移 {(peak_idx_zero - peak_idx_raw)} 采样点)")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t * 1000, mixed, linewidth=1.0, color="#1f77b4", label="Raw (noisy)")
    axes[0].plot(t * 1000, sine, linewidth=1.5, color="black", linestyle="--", label="Clean sine (ref)")
    axes[0].set_ylabel("Amplitude", fontsize=11)
    axes[0].set_title("原始信号 + 干净正弦(参考)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend(fontsize=9, loc="upper right")

    axes[1].plot(t * 1000, filtered_normal, linewidth=1.0, color="#d62728", label="Normal FIR")
    axes[1].plot(t * 1000, sine, linewidth=1.5, color="black", linestyle="--", label="Clean sine (ref)")
    axes[1].set_ylabel("Amplitude", fontsize=11)
    axes[1].set_title(f"普通 FIR 滤波 (有 {group_delay/fs*1000:.1f}ms 相位延迟)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend(fontsize=9, loc="upper right")

    axes[2].plot(t * 1000, filtered_zero, linewidth=1.0, color="#2ca02c", label="Zero-phase (filtfilt)")
    axes[2].plot(t * 1000, sine, linewidth=1.5, color="black", linestyle="--", label="Clean sine (ref)")
    axes[2].set_ylabel("Amplitude", fontsize=11)
    axes[2].set_title("零相位滤波 (filtfilt, 相位对齐)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time (ms)", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.7)
    axes[2].legend(fontsize=9, loc="upper right")

    fig.suptitle("Phase Delay Comparison: Normal FIR vs Zero-Phase", fontsize=13, fontweight="bold")
    fig.tight_layout()

    return fig


def main():
    fs = 1000.0
    duration = 1.0

    sine_freq = 50.0
    sine_amp = 2.0
    noise_amp = 0.5

    fir_cutoff = 100.0
    fir_order = 63

    print("=" * 70)
    print("SignalLab Demo v2 - 频域分析 + 数字滤波 + 修复验证")
    print("=" * 70)

    print(f"\n[1] 生成信号: {sine_freq}Hz 正弦波 ({sine_amp}V) + 白噪声 ({noise_amp}V)")
    print(f"    采样率 {fs}Hz, 时长 {duration}s")
    t_sine, sine = sl.generate_sine(fs=fs, duration=duration, frequency=sine_freq, amplitude=sine_amp)
    t_noise, noise = sl.generate_white_noise(fs=fs, duration=duration, amplitude=noise_amp, seed=42)
    mixed = sine + noise

    print("\n" + "-" * 70)
    print("[2] 时域分析 (原始信号)")
    print("-" * 70)
    stats_raw = sl.compute_statistics(mixed)
    print(f"  均值     : {stats_raw['mean']:.6f}")
    print(f"  方差     : {stats_raw['variance']:.6f}")
    print(f"  峰值     : {stats_raw['peak']:.4f} V")
    print(f"  峰峰值   : {stats_raw['peak_to_peak']:.4f} V")
    print(f"  RMS      : {stats_raw['rms']:.4f} V")

    sl.plot_time_domain(
        t=t_sine,
        signal=mixed,
        signal_type="Raw Signal (Sine + Noise)",
        params={"fs": fs, "freq": sine_freq, "sine_amp": sine_amp, "noise_amp": noise_amp},
        show=False,
    )

    print("\n" + "-" * 70)
    print("[3] 频域分析: FFT 单边幅度谱 (默认 Hann 窗)")
    print("-" * 70)
    freqs_fft, mag_fft = sl.compute_fft(mixed, fs, window="hann", nfft=2048)
    peak_idx = np.argmax(mag_fft[1:]) + 1
    print(f"  峰值频率  : {freqs_fft[peak_idx]:.2f} Hz (理论值: {sine_freq} Hz)")
    print(f"  峰值幅度  : {mag_fft[peak_idx]:.4f} V (理论值: {sine_amp} V)")
    print(f"  频率分辨率: {fs/2048:.3f} Hz (补零到 2048 点)")

    sl.plot_fft_spectrum(
        freqs=freqs_fft,
        magnitude=mag_fft,
        signal_type="Raw Signal (Hann window)",
        params={"fs": fs, "window": "hann"},
        xlim=(0, fs / 2),
        show=False,
    )

    print("\n" + "-" * 70)
    print("[4] 功率谱密度 (Welch 法)")
    print("-" * 70)
    freqs_psd, psd = sl.compute_psd_welch(mixed, fs, nperseg=256)
    print(f"  窗长: 256, 重叠: 128")
    print(f"  50Hz 处 PSD: {psd[int(sine_freq / (fs / 256))]:.6f} V\u00b2/Hz")

    sl.plot_psd(
        freqs=freqs_psd,
        psd=psd,
        signal_type="Raw Signal",
        params={"fs": fs, "nperseg": 256},
        xlim=(0, fs / 2),
        show=False,
    )

    print("\n" + "-" * 70)
    print(f"[5] FIR 低通滤波 (零相位模式): 截止 {fir_cutoff}Hz, 阶数 {fir_order}")
    print("-" * 70)
    b = sl.fir_lowpass(cutoff=fir_cutoff, fs=fs, order=fir_order)
    print(f"  滤波器抽头数: {len(b)}")
    filtered = sl.apply_filter(mixed, b, zero_phase=True)
    print("  零相位滤波完成!")

    print("\n" + "-" * 70)
    print("[6] 滤波前后时域波形对比")
    print("-" * 70)
    sl.plot_filter_compare_time(
        t=t_sine,
        signal_before=mixed,
        signal_after=filtered,
        params={"cutoff": fir_cutoff, "order": fir_order, "zero_phase": True},
        show=False,
    )

    print("\n" + "-" * 70)
    print("[7] 滤波前后频谱对比")
    print("-" * 70)
    freqs_after, mag_after = sl.compute_fft(filtered, fs, window="hann", nfft=2048)
    sl.plot_filter_compare_freq(
        freqs_before=freqs_fft,
        mag_before=mag_fft,
        freqs_after=freqs_after,
        mag_after=mag_after,
        params={"cutoff": fir_cutoff, "order": fir_order},
        xlim=(0, fs / 2),
        show=False,
    )

    print("\n" + "-" * 70)
    print("[8] 滤波后时域统计 (联动验证)")
    print("-" * 70)
    stats_filtered = sl.compute_statistics(filtered)
    print(f"  {'指标':<10} {'滤波前':>12} {'滤波后':>12} {'变化':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    for key in ["mean", "variance", "peak", "peak_to_peak", "rms"]:
        v_before = stats_raw[key]
        v_after = stats_filtered[key]
        diff = v_after - v_before
        print(f"  {key:<10} {v_before:>12.6f} {v_after:>12.6f} {diff:>+10.6f}")

    fig_leak = demo_window_leak_fix()
    fig_phase = demo_zero_phase_fix()

    print("\n" + "=" * 70)
    print("所有图表已生成，请查看弹出窗口。")
    print("特别注意看:")
    print("  - 'Spectrum Leakage Comparison': 验证加窗抑制泄漏")
    print("  - 'Phase Delay Comparison': 验证零相位滤波对齐效果")
    print("演示结束!")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
