import matplotlib.pyplot as plt
from skimage import io, img_as_float
from scipy.ndimage import uniform_filter, gaussian_filter
import os

def show_comparison(original, avg_filtered, gauss_filtered, title_prefix=""):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title(f"{title_prefix} – oryginalny")
    axs[0].axis('off')

    axs[1].imshow(avg_filtered, cmap='gray')
    axs[1].set_title("Filtr uśredniający")
    axs[1].axis('off')

    axs[2].imshow(gauss_filtered, cmap='gray')
    axs[2].set_title("Filtr Gaussa")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

def apply_average_filter(image, size=5):
    return uniform_filter(image, size=size)

def apply_gaussian_filter(image, sigma=1):
    return gaussian_filter(image, sigma=sigma)

if __name__ == "__main__":
    paths = {
        "characters_test_pattern": r"Lista2/characters_test_pattern.tif",
        "zoneplate": r"Lista2/zoneplate.tif"
    }

    kernel_sizes = [3, 5, 10]

    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"Brakuje pliku: {path}")
            continue

        print(f"Przetwarzam obraz: {name}")
        image = img_as_float(io.imread(path, as_gray=True))

        for size in kernel_sizes:
            avg_filtered = apply_average_filter(image, size=size)
            sigma = size / 3.0
            gauss_filtered = apply_gaussian_filter(image, sigma=sigma)

            show_comparison(image, avg_filtered, gauss_filtered, title_prefix=f"{name}, maska {size}x{size}")
