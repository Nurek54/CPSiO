import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
from skimage.filters.rank import equalize
from skimage.morphology import disk
from skimage.exposure import equalize_adapthist
import numpy as np
import os

# Funkcja: porównanie oryginału i przetworzonego obrazu
def show_comparison(original, processed, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Obraz oryginalny")
    axs[0].axis('off')

    axs[1].imshow(processed, cmap='gray')
    axs[1].set_title(title)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

# a) Lokalne wyrównywanie histogramu (wersja rank filter)
def local_hist_equalization(image, radius=15):
    image_u8 = img_as_ubyte(img_as_float(image))
    return equalize(image_u8, footprint=disk(radius))

# b) Lokalna poprawa kontrastu – CLAHE (adaptive histogram equalization)
def adaptive_contrast_enhancement(image, clip_limit=0.03, kernel_size=15):
    return equalize_adapthist(img_as_float(image), clip_limit=clip_limit, kernel_size=kernel_size)

# Główna część programu
if __name__ == "__main__":
    # Ścieżka do pliku testowego (dostosuj jeśli potrzeba)
    path = r"hidden-symbols.tif"

    if not os.path.exists(path):
        print("❌ Plik nie istnieje:", path)
    else:
        # Wczytanie obrazu
        image = io.imread(path, as_gray=True)

        # a) Lokalne wyrównywanie histogramu
        local_eq = local_hist_equalization(image, radius=15)
        show_comparison(image, local_eq, "Lokalne wyrównywanie histogramu (r=15)")

        # b) Lokalna poprawa kontrastu – CLAHE
        local_clahe = adaptive_contrast_enhancement(image, clip_limit=0.03, kernel_size=15)
        show_comparison(image, local_clahe, "Lokalna poprawa kontrastu (CLAHE)")
