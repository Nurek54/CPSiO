import numpy as np
import matplotlib.pyplot as plt

def generate_sine(freq, time, length):
    x = np.linspace(0, time, length) # Utworzenie dokładnie length punktów w przedziale [0, time)
    y = np.sin(2 * np.pi * freq * x) # Utworzenie sygnału sinusoidalnego o danej częstotliwości dla przedziału x

    return x, y

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

def run(frequencies, time, length):
    signals_generated = []
    for f in frequencies:
        signals_generated.append(generate_sine(freq=f, time=time, length=length))

    signal_overlaped = np.sum([y for x, y in signals_generated], axis=0)
    domain_x = signals_generated[0][0] 

    fft_result = generate_fft(signal_overlaped, time / length)

    inv_fft_result = generate_ifft(fft_result["y"])

    plt.figure(figsize=(10, 10))
    title = "Frequences given as parameter: " + "& ".join(f"{f} Hz" for f in frequencies)
    plt.suptitle(title)

    plt.subplot(3, 1, 1)
    plt.plot(domain_x, signal_overlaped)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Signal in time domain")

    plt.subplot(3, 1, 2)
    plt.plot(fft_result["x"], np.abs(fft_result["y"]) / fft_result["n"])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("Signal in frequency domain (FFT)")

    plt.subplot(3, 1, 3)
    plt.plot(domain_x, inv_fft_result)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Signal after Inverse Fourier Transform")

    plt.tight_layout()
    plt.show()


run([50], 1, 65536)
