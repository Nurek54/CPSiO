import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from skimage import io, img_as_float, filters
from scipy.ndimage import gaussian_laplace
import numpy as np
import os

# Funkcja do wyświetlania wyników
def show_edges(original, edges_dict, title_prefix=""):
    num_filters = len(edges_dict)
    fig, axs = plt.subplots(1, num_filters + 1, figsize=(5 * (num_filters + 1), 5))

    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Obraz oryginalny")
    axs[0].axis('off')

    for i, (name, edge_img) in enumerate(edges_dict.items(), start=1):
        axs[i].imshow(edge_img, cmap='gray')
        axs[i].set_title(name)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

# Filtry gradientowe
def apply_gradient_filters(image):
    edge_sobel = filters.sobel(image)
    edge_prewitt = filters.prewitt(image)
    edge_roberts = filters.roberts(image)
    edge_log = gaussian_laplace(image, sigma=1)

    return {
        "Sobel": edge_sobel,
        "Prewitt": edge_prewitt,
        "Roberts": edge_roberts,
        "LoG": np.abs(edge_log)
    }

# Główna część
if __name__ == "__main__":
    # Ścieżka do obrazu testowego
    path = r"testpat1.png"

    if not os.path.exists(path):
        print(f"❌ Brakuje pliku: {path}")
    else:
        print(f"✅ Przetwarzam: {path}")
        image = img_as_float(io.imread(path, as_gray=True))

        edges = apply_gradient_filters(image)
        show_edges(image, edges, title_prefix="Krawędzie")
