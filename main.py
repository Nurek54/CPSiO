import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

def load_ecg(file_path):
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

    return time, signal


def plot_ecg(time, signal, start_time=0, end_time=5):
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


# Podaj poprawną ścieżkę do pliku!
file_path = "C:/Users/g_sie/OneDrive/Pulpit/CPSiO/ekg100.txt"
time, signal = load_ecg(file_path)

plot_ecg(time, signal, start_time=0, end_time=5)
