# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:02:02 2023

@author: heng.zhang

"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def virtual_bass_phase_vocoder(input_file, output_file, cutoff_freq=100.0, gain_h2=0.5, gain_h3=0.25):
    """
    基于Phase Vocoder变调的虚拟低音增强
    
    Args:
        input_file: 输入音频路径
        output_file: 输出音频路径
        cutoff_freq: 低音截止频率 (Hz)，提取该频率以下的信号生成谐波
        gain_h2: 二次谐波（+1八度）的增益
        gain_h3: 三次谐波（+1八度+纯五度）的增益
    """
    # 1. 加载音频
    y, sr = librosa.load(input_file, sr=None)
    
    # 2. 提取低频分量 (Bass Component)
    # 使用低通滤波器提取需要增强的低音部分
    y_bass = lowpass_filter(y, cutoff_freq, sr)
    
    # 3. 生成谐波 (基于 Phase Vocoder 的 Pitch Shifting)
    # librosa.effects.pitch_shift 内部使用 phase vocoder 实现
    
    print("正在生成二次谐波 (Phase Vocoder Pitch Shift +12 semitones)...")
    # 向上变调12个半音 (1个八度) -> 频率 x2 (2nd Harmonic)
    y_h2 = librosa.effects.pitch_shift(y_bass, sr=sr, n_steps=12)
    
    print("正在生成三次谐波 (Phase Vocoder Pitch Shift +19 semitones)...")
    # 向上变调19个半音 (1个八度 + 纯五度 7半音) -> 频率 x3 (3rd Harmonic)
    # 注：严格来说x3是+19.02半音，这里近似为19
    y_h3 = librosa.effects.pitch_shift(y_bass, sr=sr, n_steps=19)
    
    # 4. 信号对齐与混合
    # 由于滤波和变调可能引入相位延迟或长度微小变化，确保长度一致
    min_len = min(len(y), len(y_h2), len(y_h3))
    y = y[:min_len]
    y_h2 = y_h2[:min_len]
    y_h3 = y_h3[:min_len]
    
    # 5. 混合 (原始信号 + 谐波)
    # 通常为了保护扬声器，原始信号可能会经过高通滤波滤除极低频，
    # 但这里为了演示"增强"，我们直接叠加到原信号上。
    y_enhanced = y + (y_h2 * gain_h2) + (y_h3 * gain_h3)
    
    # 6. 防止削波 (Normalization)
    max_val = np.max(np.abs(y_enhanced))
    if max_val > 1.0:
        y_enhanced = y_enhanced / max_val
        print("已应用归一化防止削波")
        
    # 7. 保存输出
    sf.write(output_file, y_enhanced, sr)
    print(f"处理完成，已保存至: {output_file}")

# 使用示例
if __name__ == "__main__":

    input_path = "./source/-12dB_20hz_20khz.wav" 
    output_path = "output_vb_enhanced.wav"
    
    virtual_bass_phase_vocoder(input_path, output_path, cutoff_freq=120)
