# -*- coding: utf-8 -*-
"""
Created on Tus Mar 2 15:27:51 2023

@author: heng.zhang

"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import soundfile as sf

def phase_vocoder_virtual_bass(audio, sr, cutoff_freq=120.0, harmonics=[2, 3, 4], gain=1.0):
    """
    基于相位声码器原理（相位倍增）生成虚拟低音谐波
    
    Args:
        audio: 输入音频数组 (1D numpy array)
        sr: 采样率
        cutoff_freq: 低音截止频率 (Hz)，低于此频率的信号将用于生成谐波
        harmonics: 要生成的谐波列表 (例如 [2, 3] 代表生成二次和三次谐波)
        gain: 谐波增强的增益
        
    Returns:
        enhanced_audio: 增强后的音频
    """
    
    # 1. STFT 参数设置
    n_fft = 2048
    hop_length = 512
    win_length = 2048
    window = 'hann'
    
    # 执行 STFT
    # f: 频率数组, t: 时间帧数组, Zxx: 复数频谱矩阵
    f, t, Zxx = scipy.signal.stft(audio, fs=sr, window=window, nperseg=win_length, noverlap=win_length-hop_length, nfft=n_fft)
    
    # 获取幅度和相位
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # 2. 确定低频截止的 Bin 索引
    # 找到频率 <= cutoff_freq 的最大索引
    cutoff_bin = np.searchsorted(f, cutoff_freq)
    
    # 初始化谐波频谱矩阵 (全零)
    # 我们将把生成的谐波放入这个空矩阵，最后再叠加
    harmonic_Zxx = np.zeros_like(Zxx, dtype=np.complex128)
    
    # num_bins = Zxx.shape,[object Object],
    # num_frames = Zxx.shape,[object Object],
    num_bins, num_frames = Zxx.shape
    
    print(f"Generating harmonics for bass up to {cutoff_freq}Hz (Bin 0-{cutoff_bin})...")
    
    # 3. 核心过程：基于相位声码器原理生成谐波
    for h in harmonics:
        # 衰减高次谐波，使其听起来更自然 (可选)
        current_gain = gain / h  
        
        # 遍历低频区域的每一个 Bin
        for b in range(1, cutoff_bin): # 从1开始避开直流分量
            target_b = b * h
            
            # 防止目标频率超出 Nyquist 频率 (超出 FFT 范围)
            if target_b < num_bins:
                # A. 幅度搬移 (Magnitude Copying)
                # 将基频的幅度复制到谐波位置
                mag_source = magnitude[b, :]
                
                # B. 相位倍增 (Phase Multiplication) - 相位声码器的关键
                # 谐波的相位应该是基频相位的 h 倍
                # exp(j * phi * h) 等价于 exp(j * phi)^h
                phase_source = phase[b, :]
                new_phase = phase_source * h
                
                # 构建复数频谱: Magnitude * e^(j * Phase)
                harmonic_component = mag_source * np.exp(1j * new_phase)
                
                # 叠加到谐波频谱矩阵中
                harmonic_Zxx[target_b, :] += harmonic_component * current_gain

    # 4. 混合
    # 策略：保留原始信号，直接叠加生成的谐波
    # 注意：为了避免低频浑浊，通常会对原始信号做高通滤波，或者直接叠加
    # 这里我们采用直接叠加的方式
    
    final_Zxx = Zxx + harmonic_Zxx
    
    # 5. ISTFT 逆变换
    _, enhanced_audio = scipy.signal.istft(final_Zxx, fs=sr, window=window, nperseg=win_length, noverlap=win_length-hop_length, nfft=n_fft)
    
    # 长度对齐 (ISTFT 可能会导致长度微小变化)
    enhanced_audio = enhanced_audio[:len(audio)]
    
    return enhanced_audio

if __name__ == "__main__":
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 原始信号：50Hz 正弦波 + 少量噪音
    freq_fund = 50
    original_signal = 0.5 * np.sin(2 * np.pi * freq_fund * t)
    
    enhanced_signal = phase_vocoder_virtual_bass(original_signal, sr, cutoff_freq=80, harmonics=[2, 3], gain=0.8)
    
    def plot_spectrum(sig, title):
        f, Pxx = scipy.signal.periodogram(sig, sr)
        plt.semilogy(f, Pxx)
        plt.xlim(0, 300)
        plt.ylim(1e-6, 1)
        plt.title(title)
        plt.grid()

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plot_spectrum(original_signal, "Original (50Hz)")
    plt.subplot(2, 1, 2)
    plot_spectrum(enhanced_signal, "Enhanced (Should have 50, 100, 150Hz)")
    plt.tight_layout()
    
    print("处理完成。请查看频谱图确认谐波是否生成。")
    plt.show()