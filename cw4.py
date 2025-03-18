import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz
from scipy.fft import fft, fftfreq
import matplotlib

matplotlib.use("TkAgg")

# Funkcja do wczytywania sygnału EKG
def load_ekg(file_path):
    data = np.loadtxt(file_path)
    time = data[:, 0]  # Pierwsza kolumna to czas
    signal = data[:, 1]  # Druga kolumna to sygnał EKG
    fs = 1 / (time[1] - time[0])  # Obliczenie częstotliwości próbkowania
    return time, signal, fs

# Funkcja do projektowania filtru Butterwortha
def design_filter(cutoff, fs, btype, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

# Funkcja do filtracji sygnału
def apply_filter(signal, b, a):
    return filtfilt(b, a, signal)

# Funkcja do wykreślania charakterystyki częstotliwościowej filtra
def plot_filter_response(b, a, fs, title):
    w, h = freqz(b, a, worN=8000)
    freq = 0.5 * fs * w / np.pi
    plt.figure(figsize=(10, 4))
    plt.plot(freq, 20 * np.log10(abs(h)), label="Charakterystyka amplitudowa")
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Tłumienie [dB]")
    plt.title(title)
    plt.grid()
    plt.axvline(x=60, color='r', linestyle='--', label="60 Hz")  # Linia dla 60 Hz
    plt.legend()
    plt.show()

# Funkcja do wykreślania sygnału i jego widma
def plot_signal_and_spectrum(time, signal, fs, title):
    # Wykres sygnału
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, signal, label="Sygnał")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title(f"{title} - Sygnał w dziedzinie czasu")
    plt.grid()
    plt.legend()

    # Widmo sygnału
    n = len(signal)
    spectrum = np.abs(fft(signal))[:n // 2]
    frequencies = fftfreq(n, 1 / fs)[:n // 2]
    plt.subplot(2, 1, 2)
    plt.plot(frequencies, spectrum, label="Widmo")
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.title(f"{title} - Widmo sygnału")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show(block = True)

# Główna funkcja
def main():
    # 1. Wczytanie sygnału
    file_path = "ekg_noise.txt"
    time, signal, fs = load_ekg(file_path)
    plot_signal_and_spectrum(time, signal, fs, "Oryginalny sygnał EKG z zakłóceniami")

    # 2. Filtracja dolnoprzepustowa (60 Hz)
    b_low, a_low = design_filter(60, fs, btype="low")
    plot_filter_response(b_low, a_low, fs, "Charakterystyka filtru dolnoprzepustowego (60 Hz)")
    filtered_low = apply_filter(signal, b_low, a_low)
    plot_signal_and_spectrum(time, filtered_low, fs, "Sygnał po filtracji dolnoprzepustowej")

    # Różnica między sygnałem oryginalnym a po filtracji dolnoprzepustowej
    difference_low = signal - filtered_low
    plot_signal_and_spectrum(time, difference_low, fs, "Różnica po filtracji dolnoprzepustowej")

    # 3. Filtracja górnoprzepustowa (5 Hz)
    b_high, a_high = design_filter(5, fs, btype="high")
    plot_filter_response(b_high, a_high, fs, "Charakterystyka filtru górnoprzepustowego (5 Hz)")
    filtered_high = apply_filter(filtered_low, b_high, a_high)
    plot_signal_and_spectrum(time, filtered_high, fs, "Sygnał po filtracji górnoprzepustowej")

    # Różnica między sygnałem po filtracji dolnoprzepustowej a po filtracji górnoprzepustowej
    difference_high = filtered_low - filtered_high
    plot_signal_and_spectrum(time, difference_high, fs, "Różnica po filtracji górnoprzepustowej")

if __name__ == "__main__":
    main()