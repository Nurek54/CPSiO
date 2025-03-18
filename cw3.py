import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

def load_ekg(file_path):
    """Wczytuje sygnał EKG, obsługując różne formaty plików."""
    data = np.loadtxt(file_path)

    if len(data.shape) == 1:  # Jeśli mamy tylko jedną kolumnę
        data = data.reshape(-1, 1)

    if data.shape[1] == 2:  # Pliki z czasem (ekg_noise.txt)
        time = data[:, 0]
        signal = data[:, 1]
    else:  # Pliki z jedną kolumną (np. ekg100.txt)
        fs = 360 if '100' in file_path else 1000  # Dopasowanie fs
        time = np.arange(len(data)) / fs
        signal = data[:, 0]

    return time, signal, fs

def plot_ekg(time, signal, start_time=0, end_time=5):
    """Wyświetla fragment sygnału EKG."""
    mask = (time >= start_time) & (time <= end_time)
    plt.figure(figsize=(10, 4))
    plt.plot(time[mask], signal[mask], label="EKG")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Wizualizacja sygnału EKG")
    plt.legend()
    plt.grid()
    plt.show(block=True)  # Wymuszenie blokowania w PyCharm

def plot_spectrum(frequencies, spectrum, fs):
    """Wyświetla widmo amplitudowe sygnału."""
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, np.abs(spectrum[:len(frequencies)]))
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.title("Widmo amplitudowe sygnału EKG")
    plt.grid()
    plt.show(block=True)

def compare_signals(original_signal, reconstructed_signal):
    """Porównuje oryginalny sygnał z odtworzonym."""
    plt.figure(figsize=(10, 4))
    plt.plot(original_signal, label="Oryginalny sygnał")
    plt.plot(reconstructed_signal, label="Odtworzony sygnał")
    plt.xlabel("Próbki")
    plt.ylabel("Amplituda")
    plt.title("Porównanie oryginalnego i odtworzonego sygnału EKG")
    plt.legend()
    plt.grid()
    plt.show(block=True)

def main():
    file_path = "C:/Users/g_sie/OneDrive/Pulpit/CPSiO/ekg100.txt"
    time, signal, fs = load_ekg(file_path)

    # 1. Wizualizacja sygnału EKG
    plot_ekg(time, signal, start_time=0, end_time=5)

    # 2. Obliczenie FFT i wyświetlenie widma amplitudowego
    fft_signal = np.fft.fft(signal)  # Szybka transformata Fouriera
    frequencies = np.fft.fftfreq(len(signal), 1/fs)
    positive_frequencies = frequencies[:len(frequencies)//2]  # Tylko dodatnie częstotliwości
    positive_spectrum = fft_signal[:len(frequencies)//2]  # Tylko dodatnie częstotliwości
    plot_spectrum(positive_frequencies, positive_spectrum, fs)

    # 3. Obliczenie odwrotnej FFT i porównanie sygnałów
    ifft_signal = np.fft.ifft(fft_signal).real  # Odwrotna FFT (bierzemy część rzeczywistą)
    compare_signals(signal, ifft_signal)

    # Obliczenie różnicy między sygnałami
    difference = signal - ifft_signal
    plt.figure(figsize=(10, 4))
    plt.plot(difference, label="Różnica sygnałów")
    plt.xlabel("Próbki")
    plt.ylabel("Amplituda")
    plt.title("Różnica między oryginalnym a odtworzonym sygnałem EKG (po IDFT)")
    plt.legend()
    plt.grid()
    plt.show(block=True)

if __name__ == "__main__":
    main()