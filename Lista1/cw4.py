import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz

def load_ekg_noise(file_path):
    data = np.loadtxt(file_path)
    time = data[:, 0]
    signal = data[:, 1]
    # Jeśli plik ekg_noise.txt jest próbkowany z fs=360,
    # to możemy to wyznaczyć z różnic w kolumnie czasu lub założyć z góry.
    # Tutaj odczytamy z "time".
    # fs ≈ 1 / (time[1] - time[0]) (zakładamy stały krok)
    fs_approx = 1.0 / (time[1] - time[0])
    return time, signal, fs_approx

def plot_time(time, signal, title, start=0, end=2):
    # Rysujemy w dziedzinie czasu wycinek [start, end] sekund
    mask = (time >= start) & (time <= end)
    plt.figure(figsize=(10,4))
    plt.plot(time[mask], signal[mask], label="sygnał")
    plt.title(title)
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_fft(signal, fs, title="Widmo sygnału"):
    N = len(signal)
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/fs)
    half = N//2
    freqs_plot = freqs[:half]
    amplitude_spectrum = np.abs(spectrum) * 2.0 / N
    amplitude_plot = amplitude_spectrum[:half]

    plt.figure(figsize=(10,4))
    plt.plot(freqs_plot, amplitude_plot)
    plt.title(title)
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.xlim([0, fs/2])
    plt.grid(True)
    plt.show()

def design_butter_lowpass(fc, fs, order=4):
    # Butterworth low-pass filter: fc=częstotliwość graniczna
    # Wn jest znormalizowane => Wn = fc / (fs/2)
    Wn = fc / (fs/2)
    b, a = butter(order, Wn, btype='low', analog=False)
    return b, a

def design_butter_highpass(fc, fs, order=4):
    Wn = fc / (fs/2)
    b, a = butter(order, Wn, btype='high', analog=False)
    return b, a

def plot_filter_response(b, a, fs, title="Charakterystyka filtra"):
    w, h = freqz(b, a, worN=1024)
    freqs = w * fs/(2*np.pi)
    gain = 20 * np.log10(np.abs(h))
    plt.figure(figsize=(10,4))
    plt.plot(freqs, gain)
    plt.title(title)
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Wzmocnienie [dB]")
    plt.grid(True)
    plt.show()

def main():
    file_path = r"ekg_noise.txt"
    time, noisy_signal, fs = load_ekg_noise(file_path)
    print(f"Szacowana fs = {fs:.2f} Hz")

    # 1) Sygnał z zakłóceniami
    print("1) Obejrzenie surowego sygnału (ekg_noise).")
    # Rysujemy np. fragment 2 sekund
    plot_time(time, noisy_signal, "Surowy sygnał EKG (z zakłóceniami)", start=0, end=2)
    # Rysujemy widmo surowego sygnału (charakterystyka amplitudowa)
    plot_fft(noisy_signal, fs, title="Widmo amplitudowe sygnału surowego")

    # 2) Filtr dolnoprzepustowy fc=60 Hz
    print("2) Filtr dolnoprzepustowy Butterworth, fc=60 Hz.")
    b_lp, a_lp = design_butter_lowpass(fc=60, fs=fs, order=4)
    print("   - Charakterystyka filtra (lowpass 60 Hz)")
    plot_filter_response(b_lp, a_lp, fs, title="Charakterystyka filtra dolnoprzepustowego (60 Hz)")
    # Filtracja
    lp_signal = filtfilt(b_lp, a_lp, noisy_signal)
    # Rysunek czasowy + widmo
    plot_time(time, lp_signal, "Sygnał po filtrze dolnoprzepustowym (60 Hz)", start=0, end=2)
    plot_fft(lp_signal, fs, "Widmo po filtrze dolnoprzepustowym (60 Hz)")
    # Różnica sygnałów
    diff_lp = noisy_signal - lp_signal
    plot_time(time, diff_lp, "Różnica (oryginał - LP) w dziedzinie czasu", start=0, end=2)
    plot_fft(diff_lp, fs, "Widmo różnicy (oryginał - LP)")

    # 3) Filtr górnoprzepustowy fc=5 Hz (na sygnale już przefiltrowanym w punkcie 2)
    print("3) Filtr górnoprzepustowy Butterworth, fc=5 Hz, na sygnale z (2).")
    b_hp, a_hp = design_butter_highpass(fc=5, fs=fs, order=4)
    print("   - Charakterystyka filtra (highpass 5 Hz)")
    plot_filter_response(b_hp, a_hp, fs, title="Charakterystyka filtra górnoprzepustowego (5 Hz)")
    # Filtracja
    bp_signal = filtfilt(b_hp, a_hp, lp_signal)
    # Wykresy
    plot_time(time, bp_signal, "Sygnał po filtrze górnoprzepustowym (5 Hz)", start=0, end=2)
    plot_fft(bp_signal, fs, "Widmo po filtrze (5 Hz) - finalne")
    # Różnica po HP
    diff_hp = lp_signal - bp_signal
    plot_time(time, diff_hp, "Różnica (LP - BP) w dziedzinie czasu", start=0, end=2)
    plot_fft(diff_hp, fs, "Widmo różnicy (LP - BP)")

    print("Gotowe! :P ")

if __name__ == "__main__":
    main()
