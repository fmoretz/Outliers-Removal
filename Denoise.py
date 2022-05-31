import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import tkinter
from tkinter import filedialog

# set up data
tkinter.Tk().withdraw()

#choose file to open
file_path = filedialog.askopenfilename()

# load data
df = pd.read_csv(str(file_path), sep=';')
df = df.dropna()

# Clean data
ylabels = df.columns

# reset index
df = df.reset_index(drop=True)

# convert dataframe to array
time_index = df.iloc[:, 0].values
sensor = df.iloc[:, 2].values

# compute fft of sensor data
dt = 0.001

n    = len(sensor)
fhat = np.fft.fft(sensor, n)
PSD  = fhat*np.conj(fhat)/n
freq = (1/(dt*n)) * np.arange(n)
L    = np.arange(1, np.floor(n/2), dtype='int')

# auto selection of frequency peaks
PSD_split = np.array_split(PSD, int(np.floor(n/50)))

## activate for debugging
plt.figure(0)
plt.tight_layout()
for i in range(0, len(PSD_split)):
    plt.subplot(1,len(PSD_split),i+1)
    plt.plot(PSD_split[i])

peak_index = np.empty(len(PSD_split))
freq_peak  = np.empty(len(PSD_split))
for i in range(len(PSD_split)):
    peak_index[i] = np.argmax(PSD_split[i])

peak_index = peak_index[1:]
for i in range(len(peak_index)):
    freq_peak[i] = PSD[int(peak_index[i])]

max_freq     = max(freq_peak)
norm_freq    = freq_peak/max_freq
rel_err_freq = np.abs(max(norm_freq) - norm_freq) * 100 
peak_passed  = []
peak_passed.append(max_freq)

for i in range(len(rel_err_freq)):
    if rel_err_freq[i] == 0.0:
        pass
    elif rel_err_freq[i] < 50:
        peak_passed.append(freq_peak[i])
    
indices   = PSD > min(peak_passed)
PSD_clean = PSD * indices
fhat_new  = fhat * indices
ffilt     = np.fft.ifft(fhat_new)

plt.figure(1)
plt.subplot(311)
plt.plot(time_index, sensor, 'k', label='Raw Signal')
plt.plot(time_index, np.real(ffilt), 'r', label='Filtered Signal')
plt.xlim(time_index[0], time_index[-1])
plt.ylabel('Sensor signal')
plt.legend()

plt.subplot(312)
plt.plot(time_index, np.real(ffilt), 'r', label='Filtered Signal')
plt.xlabel('time index')
plt.xlim(time_index[0], time_index[-1])
plt.ylabel('Sensor signal')
plt.legend()


plt.subplot(313)
plt.plot(freq[L], PSD[L], 'k', label='Raw PSD')
plt.plot(freq[L], PSD_clean[L], 'r', label='Clean PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.xlim(freq[L[0]], freq[L[-1]])


plt.show()