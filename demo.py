import signallab as sl


def main():
    fs = 1000.0
    duration = 1.0

    sine_freq = 50.0
    sine_amp = 2.0
    noise_amp = 0.5

    fir_cutoff = 100.0
    fir_order = 63

    print("=" * 70)
    print("SignalLab Demo: Frequency Analysis + FIR Filtering")
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
    print("[3] 频域分析: FFT 单边幅度谱")
    print("-" * 70)
    freqs_fft, mag_fft = sl.compute_fft(mixed, fs)
    peak_idx = int(sine_freq / (fs / len(mixed)))
    print(f"  频谱分辨率: {fs/len(mixed):.2f} Hz")
    print(f"  50Hz 处幅度: {mag_fft[peak_idx]:.4f} V (理论值: {sine_amp:.4f} V)")

    sl.plot_fft_spectrum(
        freqs=freqs_fft,
        magnitude=mag_fft,
        signal_type="Raw Signal",
        params={"fs": fs},
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
    print(f"[5] FIR 低通滤波: 截止 {fir_cutoff}Hz, 阶数 {fir_order}")
    print("-" * 70)
    b = sl.fir_lowpass(cutoff=fir_cutoff, fs=fs, order=fir_order)
    print(f"  滤波器抽头数: {len(b)}")
    filtered = sl.apply_filter(mixed, b)
    print("  滤波完成!")

    print("\n" + "-" * 70)
    print("[6] 滤波前后时域波形对比")
    print("-" * 70)
    sl.plot_filter_compare_time(
        t=t_sine,
        signal_before=mixed,
        signal_after=filtered,
        params={"cutoff": fir_cutoff, "order": fir_order},
        show=False,
    )

    print("\n" + "-" * 70)
    print("[7] 滤波前后频谱对比")
    print("-" * 70)
    freqs_after, mag_after = sl.compute_fft(filtered, fs)
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

    print("\n" + "=" * 70)
    print("所有图表已生成，请查看弹出窗口。")
    print("演示结束!")
    print("=" * 70)

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
