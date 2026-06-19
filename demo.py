import signallab as sl


def main():
    fs = 1000.0
    duration = 1.0

    sine_freq = 50.0
    sine_amp = 2.0
    noise_amp = 0.5

    print("=" * 60)
    print("SignalLab Demo: Sine Wave + White Noise")
    print("=" * 60)

    t_sine, sine = sl.generate_sine(
        fs=fs,
        duration=duration,
        frequency=sine_freq,
        amplitude=sine_amp,
    )
    print(f"\n[1] 生成正弦波: {sine_freq}Hz, 幅度 {sine_amp}V")
    sine_stats = sl.compute_statistics(sine)
    print(f"    统计: 均值={sine_stats['mean']:.4f}, RMS={sine_stats['rms']:.4f}V")

    t_noise, noise = sl.generate_white_noise(
        fs=fs,
        duration=duration,
        amplitude=noise_amp,
        seed=42,
    )
    print(f"\n[2] 生成白噪声: 幅度 {noise_amp}V (固定随机种子)")
    noise_stats = sl.compute_statistics(noise)
    print(f"    统计: 均值={noise_stats['mean']:.4f}, RMS={noise_stats['rms']:.4f}V")

    mixed = sine + noise
    print(f"\n[3] 叠加信号: 正弦波 + 白噪声")
    mixed_stats = sl.compute_statistics(mixed)
    print(f"    时域统计结果:")
    print(f"      均值     : {mixed_stats['mean']:.6f}")
    print(f"      方差     : {mixed_stats['variance']:.6f}")
    print(f"      峰值     : {mixed_stats['peak']:.4f} V")
    print(f"      峰峰值   : {mixed_stats['peak_to_peak']:.4f} V")
    print(f"      RMS 有效值: {mixed_stats['rms']:.4f} V")

    params = {
        "fs": fs,
        "freq": sine_freq,
        "sine_amp": sine_amp,
        "noise_amp": noise_amp,
    }
    sl.plot_time_domain(
        t=t_sine,
        signal=mixed,
        signal_type="Sine + White Noise",
        params=params,
        show=True,
    )

    print("\n[4] 波形图已绘制完成")
    print("\n演示结束!")


if __name__ == "__main__":
    main()
