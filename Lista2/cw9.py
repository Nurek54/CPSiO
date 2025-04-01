import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from skimage import io, img_as_float
from scipy.ndimage import uniform_filter, median_filter, minimum_filter, maximum_filter
import os

# Funkcja pomocnicza: porównanie przed/po
def show_filter_result(original, filtered, title):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Obraz oryginalny")
    axs[0].axis('off')

    axs[1].imshow(filtered, cmap='gray')
    axs[1].set_title(title)
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

# a) Filtr uśredniający (mean)
def apply_mean_filter(image, size=3):
    return uniform_filter(image, size=size)

# b) Filtr medianowy
def apply_median_filter(image, size=3):
    return median_filter(image, size=size)

# c) Filtr minimum
def apply_min_filter(image, size=3):
    return minimum_filter(image, size=size)

# c) Filtr maksimum
def apply_max_filter(image, size=3):
    return maximum_filter(image, size=size)

# Ścieżki do plików – dostosuj do swojej lokalizacji
paths = {
    "pepper": r"cboard_pepper_only.tif",
    "salt": r"cboard_salt_only.tif",
    "salt_pepper": r"cboard_salt_pepper.tif"
}

# Przetwarzanie każdego obrazu
for name, path in paths.items():
    if not os.path.exists(path):
        print(f"❌ Brakuje pliku: {path}")
        continue

    print(f"✅ Przetwarzam: {name}")
    image = img_as_float(io.imread(path, as_gray=True))

    # Mean filter
    result_mean = apply_mean_filter(image, size=3)
    show_filter_result(image, result_mean, f"{name} – filtr uśredniający 3x3")

    # Median filter
    result_median = apply_median_filter(image, size=3)
    show_filter_result(image, result_median, f"{name} – filtr medianowy 3x3")

    # Min filter
    result_min = apply_min_filter(image, size=3)
    show_filter_result(image, result_min, f"{name} – filtr minimum 3x3")

    # Max filter
    result_max = apply_max_filter(image, size=3)
    show_filter_result(image, result_max, f"{name} – filtr maksimum 3x3")
