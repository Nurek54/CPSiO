import matplotlib.pyplot as plt
from skimage import io, exposure, img_as_float
import numpy as np

# Funkcja pomocnicza do konwersji obrazu do zakresu [0, 1]
def to_float(image):
    return img_as_float(image)

# Wyrównywanie histogramu (globalne)
def equalize_histogram(image):
    return exposure.equalize_hist(to_float(image))

# Wizualizacja: obraz i jego histogram przed i po
def show_histogram_comparison(original, equalized, title=""):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title("Obraz oryginalny")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(equalized, cmap='gray')
    axs[0, 1].set_title("Po wyrównaniu histogramu")
    axs[0, 1].axis('off')

    axs[1, 0].hist(original.ravel(), bins=256, range=(0, 1), color='gray')
    axs[1, 0].set_title("Histogram oryginalny")

    axs[1, 1].hist(equalized.ravel(), bins=256, range=(0, 1), color='gray')
    axs[1, 1].set_title("Histogram po wyrównaniu")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Przykład użycia:
if __name__ == "__main__":
    # Ścieżki do testowych plików (dostosuj do siebie)
    paths = {
        "pollen-dark": r"C:\Users\g_sie\OneDrive\Pulpit\CPSiO\pollen-dark.tif",
        "pollen-ligt": r"C:\Users\g_sie\OneDrive\Pulpit\CPSiO\pollen-ligt.tif",
        "pollen-lowcontrast": r"C:\Users\g_sie\OneDrive\Pulpit\CPSiO\pollen-lowcontrast.tif",
        "spectrum": r"C:\Users\g_sie\OneDrive\Pulpit\CPSiO\spectrum.tif",
        "pout": r"C:\Users\g_sie\OneDrive\Pulpit\CPSiO\pout.tif",
        "chest-xray": r"C:\Users\g_sie\OneDrive\Pulpit\CPSiO\chest-xray.tif"
    }

    for name, path in paths.items():
        img = io.imread(path, as_gray=True)
        img_eq = equalize_histogram(img)
        show_histogram_comparison(to_float(img), img_eq, f"Wyrównywanie histogramu – {name}")
