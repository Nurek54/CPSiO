import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def load_ekg(file_path, fs=None, channel=0):
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if fs is None:
        if '100' in file_path:
            fs = 360
        else:
            fs = 1000

    if data.shape[1] == 2:
        # Plik ma czas + amplitudę
        time = data[:, 0]
        if channel == 0 or channel == 1:
            signal = data[:, 1]
        else:
            raise ValueError(f"Plik {file_path} ma tylko czas i amplitudę. Dostępny channel=0 lub 1.")
    else:
        # Plik ma tylko sygnały (wiele kanałów)
        if channel >= data.shape[1]:
            raise ValueError(
                f"Wybrany channel={channel}, ale plik ma tylko {data.shape[1]} kanałów (indeksy 0 do {data.shape[1] - 1}).")
        signal = data[:, channel]
        n_samples = len(signal)
        time = np.arange(n_samples) / fs

    print(f"Wczytano plik: {file_path}")
    print(f"Liczba kanałów w pliku: {data.shape[1]}")
    print(f"Używany channel: {channel}")
    return time, signal


def plot_ekg(time, signal, start_time=0.0, end_time=5.0):
    mask = (time >= start_time) & (time <= end_time)
    plt.figure(figsize=(10, 4))
    plt.plot(time[mask], signal[mask])
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title(f"Sygnał EKG od {start_time}s do {end_time}s")
    plt.grid(True)
    plt.show()


def save_signal(time, signal, start_time, end_time, output_file):
    mask = (time >= start_time) & (time <= end_time)
    clipped_data = np.column_stack((time[mask], signal[mask]))
    np.savetxt(output_file, clipped_data, header="Czas[s] Amplituda", comments='')
    print(f"Zapisano wycinek sygnału do pliku: {output_file}")


if __name__ == "__main__":
    file_path = r"ekg1.txt"
    channel = 0
    start_time = 1.0  # <- czas początkowy wycinka
    end_time = 2.0  # <- czas końcowy wycinka

    time, signal = load_ekg(file_path, channel=channel)
    plot_ekg(time, signal, start_time=start_time, end_time=end_time)
    save_signal(time, signal, start_time=start_time, end_time=end_time, output_file="ekg_cut.txt")
