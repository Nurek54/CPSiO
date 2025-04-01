import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from skimage import io, img_as_float
from scipy.ndimage import laplace, gaussian_filter
import numpy as np
import os

# Funkcja do wyświetlania wyników
def show_sharpening(original, highpass, sharpened, title=""):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Obraz oryginalny")
    axs[0].axis('off')

    axs[1].imshow(highpass, cmap='gray')
    axs[1].set_title("Obraz górnoprzepustowy")
    axs[1].axis('off')

    axs[2].imshow(sharpened, cmap='gray')
    axs[2].set_title("Po wyostrzeniu")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

# a) Filtr górnoprzepustowy (Laplace)
def highpass_laplace(image):
    return laplace(image)

# b) Wyostrzenie przez odejmowanie rozmycia Gaussa
def sharpen_with_blur_subtraction(image, sigma=1.0):
    blurred = gaussian_filter(image, sigma=sigma)
    highpass = image - blurred
    sharpened = np.clip(image + highpass, 0, 1)
    return highpass, sharpened

# Główna część programu
if __name__ == "__main__":
    # Ścieżka do testowego obrazu
    path = r"blurry-moon.tif"

    if not os.path.exists(path):
        print(f"❌ Brakuje pliku: {path}")
    else:
        print(f"✅ Przetwarzam: {path}")
        image = img_as_float(io.imread(path, as_gray=True))

        # a) Laplace jako filtr górnoprzepustowy
        highpass = highpass_laplace(image)
        sharpened_a = np.clip(image - highpass, 0, 1)  # czasem -laplace

        show_sharpening(image, highpass, sharpened_a, title="Wyostrzanie – Laplace")

        # b) Wyostrzanie przez odjęcie rozmycia Gaussa
        highpass_b, sharpened_b = sharpen_with_blur_subtraction(image, sigma=1.0)
        show_sharpening(image, highpass_b, sharpened_b, title="Wyostrzanie – odejmowanie rozmycia")
