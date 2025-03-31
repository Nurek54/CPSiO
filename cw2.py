import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def generate_signal(freq_list, fs, n_samples):
    t = np.arange(n_samples) / fs
    signal = np.zeros_like(t)
    for f in freq_list:
        signal += np.sin(2 * np.pi * f * t)
    return t, signal

def plot_signal(t, signal, title="Sygnał w dziedzinie czasu", start_time=0, end_time=None):
    if end_time is None:
        end_time = t[-1]
    mask = (t >= start_time) & (t <= end_time)
    plt.figure(figsize=(10, 4))
    plt.plot(t[mask], signal[mask], label="sygnał")
    plt.title(title)
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.legend()
    plt.show()

def compute_and_plot_fft(signal, fs, title="Widmo sygnału"):
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
    plt.xlim([0, fs/2])
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.show()
    return freqs, spectrum

def compare_with_ifft(signal, spectrum, t=None, show_plot=False, title="Porównanie oryginał vs ifft"):
    reconstructed = np.fft.ifft(spectrum).real
    mse = np.mean((signal - reconstructed)**2)
    print(f"Błąd odwrotnej FFT (MSE): {mse:e}")
    if show_plot and t is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal, label="Oryginalny")
        plt.plot(t, reconstructed, '--', label="Odtworzony (ifft)")
        plt.title(title)
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda")
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 0.01)
        plt.show()
    return reconstructed

def main():
    fs = 1000
    N = 65536
    f1 = 50
    f2 = 60
    print("ĆWICZENIE 2 — TRANSFORMATA FOURIERA")

    print("\n[1] Generujemy sin(50Hz)")
    t, sig_50 = generate_signal([f1], fs, N)
    plot_signal(t, sig_50, title="Czysta sinusoida 50Hz (fragment)", start_time=0, end_time=0.01)

    print("[2] Widmo amplitudowe sin(50Hz)")
    freqs, spectrum = compute_and_plot_fft(sig_50, fs)
    reconstructed_50 = compare_with_ifft(sig_50, spectrum, t=t, show_plot=True, title="IFFT – czysta sinusoida")

    print("\n[3] Generujemy sin(50Hz) + sin(60Hz)")
    t2, sig_50_60 = generate_signal([f1, f2], fs, N)
    plot_signal(t2, sig_50_60, title="Mieszanina 50Hz + 60Hz (fragment)", start_time=0, end_time=0.01)

    print("Widmo amplitudowe (50Hz + 60Hz)")
    freqs_2, spectrum_2 = compute_and_plot_fft(sig_50_60, fs)
    reconstructed_50_60 = compare_with_ifft(sig_50_60, spectrum_2, t=t2, show_plot=True, title="IFFT – mieszanina 50Hz + 60Hz")

    print("\n[4] Różne fs (i różne czasy trwania).")
    fs_values = [500, 1000, 2000]
    for fs_test in fs_values:
        print(f"\nfs = {fs_test} Hz")
        t3, sig_mix = generate_signal([50, 60], fs_test, 65536)
        compute_and_plot_fft(sig_mix, fs_test)

    print("\n[5] Odwrotna FFT sprawdzona w compare_with_ifft().")

if __name__ == "__main__":
    main()
