# -*- coding: utf-8 -*-
"""
Created on Tus Dec 16 15:06:15 2021

@author: heng.zhang

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write



def audio_data(amp, freq, sample_rate, duration):
    
    # 生成音频信号
    t = np.linspace(0, duration, int(duration*sample_rate))
    audio = amp*np.sin(2 * np.pi * freq * t)
    
    return audio
    
def noise_data(amp, sample_rate, duration):
    
    # 添加噪声(duration * sampling_freq个(0,1]之间的值)
    noise = amp * np.random.rand(int(duration*sample_rate))
    
    return noise
    
    

if __name__ == "__main__":
    
    output_file = 'output_generated.wav'
    
    # 指定音频生成的参数
    duration = 0.03            # 单位秒
    sample_rate = 48000   # 单位Hz
    tone_freq = 300         # 音调的频率
    
    audio = audio_data(1, tone_freq, sample_rate, duration)
    
    noise = noise_data(0.4, sample_rate, duration)

    audio += noise

    scaling_factor = pow(2,15) - 1  # 转换为16位整型数
    audio_normalized = audio / np.max(np.abs(audio))    # 归一化
    audio_scaled = np.int16(audio_normalized * scaling_factor)  # 这句话什么意思

    write(output_file, sample_rate, audio_scaled) # 写入输出文件
    
    # audio = audio[:100]
    
    x_values = np.arange(0, len(audio), 1) / float(sample_rate)
    x_values *= 1000    # 将时间轴单位转换为秒    
    
    plt.plot(x_values, audio, color='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Audio signal')
    plt.show()
