import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Użycie backendu TkAgg
matplotlib.use("TkAgg")

# 1. Generowanie fali sinusoidalnej o częstotliwości 50 Hz i długości 65536 próbek
def generate_sine_wave(freq, sample_rate, duration, n_samples=65536):
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal

# 2. Obliczenie dyskretnej transformaty Fouriera i wyświetlenie widma amplitudowego
def plot_amplitude_spectrum(signal, sample_rate):
    n = len(signal)
    fft_result = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(n, 1 / sample_rate)

    # Widmo amplitudowe
    amplitude_spectrum = np.abs(fft_result) / n

    # Wyświetlenie widma w zakresie [0, fs/2]
    positive_freqs = fft_freqs[:n // 2]
    positive_amplitude = amplitude_spectrum[:n // 2]

    plt.figure(figsize=(10, 4))
    plt.plot(positive_freqs, positive_amplitude)
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.title("Widmo amplitudowe")
    plt.grid()
    plt.show()

# 3. Generowanie mieszaniny dwóch fal sinusoidalnych
def generate_mixed_sine_waves(freq1, freq2, sample_rate, duration, n_samples=65536):
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal1 = np.sin(2 * np.pi * freq1 * t)
    signal2 = np.sin(2 * np.pi * freq2 * t)
    mixed_signal = signal1 + signal2
    return t, mixed_signal

# 4. Powtórzenie eksperymentów dla różnych czasów trwania sygnałów i częstotliwości próbkowania
def repeat_experiment(freq1, freq2, durations, sample_rates):
    for duration in durations:
        for fs in sample_rates:
            print(f"Czas trwania sygnału: {duration} s, Częstotliwość próbkowania: {fs} Hz")

            # Generowanie sygnału z jedną częstotliwością
            t, signal = generate_sine_wave(freq1, fs, duration)
            plot_amplitude_spectrum(signal, fs)

            # Generowanie mieszaniny sygnałów
            t, mixed_signal = generate_mixed_sine_waves(freq1, freq2, fs, duration)
            plot_amplitude_spectrum(mixed_signal, fs)

# 5. Obliczenie odwrotnej transformaty Fouriera i porównanie z sygnałem oryginalnym
def compare_with_original(signal, sample_rate):
    n = len(signal)
    fft_result = np.fft.fft(signal)
    reconstructed_signal = np.fft.ifft(fft_result)

    # Porównanie sygnałów
    t = np.linspace(0, n / sample_rate, n, endpoint=False)
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, label="Oryginalny sygnał")
    plt.plot(t, reconstructed_signal.real, label="Odtworzony sygnał", linestyle='--')
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Porównanie sygnału oryginalnego z odtworzonym")
    plt.legend()
    plt.grid()
    plt.show()

# Parametry
freq1 = 50  # Częstotliwość pierwszej fali sinusoidalnej [Hz]
freq2 = 60  # Częstotliwość drugiej fali sinusoidalnej [Hz]
sample_rates = [256, 1000, 4096]  # Różne częstotliwości próbkowania [Hz]
durations = [1, 2, 5]  # Różne czasy trwania sygnałów [s]

# 1. Generowanie fali sinusoidalnej o długości 65536 próbek
t, signal = generate_sine_wave(freq1, sample_rates[0], durations[0])

# 2. Wyznaczanie widma amplitudowego
plot_amplitude_spectrum(signal, sample_rates[0])

# 3. Generowanie mieszaniny dwóch fal sinusoidalnych
t, mixed_signal = generate_mixed_sine_waves(freq1, freq2, sample_rates[0], durations[0])
plot_amplitude_spectrum(mixed_signal, sample_rates[0])

# 4. Powtórzenie eksperymentów dla różnych czasów trwania i częstotliwości próbkowania
repeat_experiment(freq1, freq2, durations, sample_rates)

# 5. Porównanie sygnałów oryginalnych z odtworzonymi
compare_with_original(signal, sample_rates[0])
compare_with_original(mixed_signal, sample_rates[0])