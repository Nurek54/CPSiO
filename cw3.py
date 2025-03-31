import numpy as np
import matplotlib.pyplot as plt

def load_signal(path, sample_frequency):
    y = np.loadtxt(path)

    length = len(y)
    time = length / sample_frequency

    x = np.linspace(0, time, length)

    return {
        "length": length,
        "time": time,
        "x": x,
        "y": y
    }

def generate_fft(signal, sample_rate):
    x = np.fft.rfftfreq(len(signal), sample_rate)
    n = len(signal) / 2
    y = np.fft.rfft(signal)

    return {
        "x": x,
        "n": n,
        "y": y
    }

def generate_ifft(signal):
    return np.fft.irfft(signal)

def run(path, sample_frequency):
    base_signal = load_signal(path=path, sample_frequency=sample_frequency)
    fft_result = generate_fft(base_signal["y"], base_signal["time"] / base_signal["length"])
    inv_fft_result = generate_ifft(fft_result["y"])

    plt.figure(figsize=(10, 10))
    title = str(path) + "analysis"
    plt.suptitle(title)

    plt.subplot(3, 1, 1)
    plt.plot(base_signal["x"], base_signal["y"])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Signal in time domain")

    plt.subplot(3, 1, 2)
    plt.plot(fft_result["x"], np.abs(fft_result["y"]) / fft_result["n"])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("Signal in frequency domain (FFT)")

    plt.subplot(3, 1, 3)
    plt.plot(base_signal["x"], inv_fft_result)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Signal after Inverse Fourier Transform")

    plt.tight_layout()
    plt.show()

run("ekg100.txt", 360)
