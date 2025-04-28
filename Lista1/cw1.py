import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def load_ekg(file_path, fs=None, channel=0):
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] == 2:
        time = data[:, 0]
        signal = data[:, 1]
    else:
        if fs is None:
            if '100' in file_path:
                fs = 360
            else:
                fs = 1000
        if channel >= data.shape[1]:
            raise ValueError("Wybrany kanał przekracza liczbę kolumn w pliku.")
        signal = data[:, channel]
        n_samples = len(signal)
        time = np.arange(n_samples) / fs
    return time, signal

def plot_ekg(time, signal, start_time=0.0, end_time=5.0):
    mask = (time >= start_time) & (time <= end_time)
    plt.figure(figsize=(10, 4))
    plt.plot(time[mask], signal[mask])
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Sygnał EKG")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "Lista1/ekg_noise.txt"
    time, signal = load_ekg(file_path, channel=2)
    plot_ekg(time, signal, start_time=1.0, end_time=2.0)
