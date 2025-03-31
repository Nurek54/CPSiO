import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def load_ekg_ecg100(file_path):
    data = np.loadtxt(file_path)
    fs = 360  # częstotliwość próbkowania dla ecg100
    # Plik ecg100.txt ma 1 kolumnę z samymi wartościami sygnału
    signal = data
    N = len(signal)
    t = np.arange(N) / fs
    return t, signal, fs

def plot_ecg_time(t, signal, title="Sygnał EKG (ecg100) - dziedzina czasu", start_time=0, end_time=2):
    mask = (t >= start_time) & (t <= end_time)
    plt.figure(figsize=(10, 4))
    plt.plot(t[mask], signal[mask])
    plt.title(title)
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.show()

def compute_and_plot_fft_ecg(signal, fs, title="Widmo amplitudowe EKG"):
    N = len(signal)
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/fs)

    half = N // 2
    freqs_plot = freqs[:half]
    amplitude_spectrum = np.abs(spectrum) * 2.0 / N
    amplitude_plot = amplitude_spectrum[:half]

    plt.figure(figsize=(10, 4))
    plt.plot(freqs_plot, amplitude_plot)
    plt.title(title)
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.xlim([0, fs/2])  # zakres [0, fs/2]
    plt.grid(True)
    plt.show()

    return spectrum, freqs

def compare_ifft(signal, spectrum):
    reconstructed = np.fft.ifft(spectrum).real
    diff = signal - reconstructed
    mse = np.mean(diff**2)
    print(f"Błąd MSE odwrotnej FFT: {mse:e}")
    return reconstructed, diff

def main():
    file_path = r"C:\Users\g_sie\OneDrive\Pulpit\CPSiO\ekg100.txt"  # dostosuj do siebie
    t, ecg_signal, fs = load_ekg_ecg100(file_path)

    # 1) Ocen sygnał EKG wizualnie
    # Wyświetlamy fragment np. 2 sekundy
    plot_ecg_time(t, ecg_signal, start_time=0, end_time=2)

    # 2) Wyznacz FFT i przedstaw widmo amplitudowe
    spectrum, freqs = compute_and_plot_fft_ecg(ecg_signal, fs)

    # 3) Odwrotna FFT i porównanie
    reconstructed, diff = compare_ifft(ecg_signal, spectrum)

    # Możesz też np. narysować porównanie oryginału i rekonstrukcji
    plt.figure(figsize=(10, 4))
    plt.plot(t[:500], ecg_signal[:500], label="Oryginalny (fragment)")
    plt.plot(t[:500], reconstructed[:500], '--', label="Rekonstrukcja IFFT (fragment)")
    plt.title("Porównanie sygnału oryginalnego i odtworzonego z IFFT")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Możesz też narysować różnicę
    plt.figure(figsize=(10, 4))
    plt.plot(t[:500], diff[:500])
    plt.title("Różnica: sygnał oryginalny - rekonstrukcja (fragment)")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
