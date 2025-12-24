# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 10:19:27 2025

@author: heng.zhang

"""

import generater as gt
import numpy as np
import matplotlib.pyplot as plt


NFFT = 512
duration = 0.05
sample_rate = 48000
frame_time = 0.01
step_time = 0.25 * frame_time

pitch_ratio = 5
ratio = 2 ** (pitch_ratio/12)

delta_ta = step_time
delta_ts = delta_ta*ratio

frame_length = int(frame_time*sample_rate)
frame_step = int(step_time*sample_rate)
num_frames = int((duration - frame_time)/step_time + 1)

audio  = gt.audio_data(0.7, 50, sample_rate, duration)
out = np.zeros(len(audio) + num_frames*int((ratio-1)*frame_time*sample_rate))


indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T


frames = audio[indices.astype(np.int32, copy=False)]

frames *= np.hamming(frame_length)

freq = np.fft.rfft(frames, NFFT)

rel_frames, img_frames = np.abs(freq), np.angle(freq)


w_bin = np.tile(np.arange(0, sample_rate/2+sample_rate/NFFT, sample_rate/NFFT), (num_frames-1, 1))

delta_w_unwrap = (img_frames[1,:] - img_frames[-1,:])/delta_ta - w_bin

delta_w_wrap = np.mod(delta_w_unwrap+np.pi, 2*np.pi)-np.pi

w_true = w_bin + delta_w_wrap

phase = np.copy(img_frames)
for i in range(1, num_frames):
    phase[i,:] = phase[i-1,:] + w_true[i-1,:]*delta_ts


out_complex = rel_frames * np.exp(1j * phase)

audio_processed = np.fft.irfft(freq)*np.hamming(NFFT)


for i in range(num_frames):
    # print(i)
    for j in range(len(indices[i,:])):
        out[indices[i,j] + i*int((ratio-1)*frame_time*sample_rate)] += audio_processed[i,j]


# mag_frames = np.absolute()   # fft的幅度(magnitude)

plt.plot(out)