import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float

# Funkcja pomocnicza – konwersja obrazu do float w zakresie [0, 1]
def to_float(image):
    return img_as_float(image)

# a) Mnożenie obrazu przez stałą
def multiply_image(image, c):
    return np.clip(to_float(image) * c, 0, 1)

# b) Transformacja logarytmiczna
def log_transform(image, c):
    return np.clip(c * np.log1p(to_float(image)), 0, 1)

# c) Transformacja kontrastu (postać sigmoidalna)
def contrast_transform(image, m=0.45, e=8):
    image_f = to_float(image)
    return 1 / (1 + (m / (image_f + 1e-5))**e)

# d) Korekcja gamma
def gamma_correction(image, c=1.0, gamma=0.4):
    return np.clip(c * (to_float(image) ** gamma), 0, 1)

# Wizualizacja przekształconego obrazu
def show_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Ścieżki do plików – dostosuj do własnej lokalizacji
path_a = r"/Lista2/chest-xray.tif"
path_b = r"/spectrum.tif"
path_c = r"/pollen-lowcontrast.tif"
path_d = r"/Lista2/aerial_view.tif"

# Wczytanie obrazów
img_a = io.imread(path_a, as_gray=True)
img_b = io.imread(path_b, as_gray=True)
img_c = io.imread(path_c, as_gray=True)
img_d = io.imread(path_d, as_gray=True)

# a) Mnożenie przez stałą
img_multiplied = multiply_image(img_a, c=1.5)
show_image(img_multiplied, "6a – Mnożenie przez stałą (c=1.5)")

# b) Transformacja logarytmiczna
img_log = log_transform(img_b, c=0.5)
show_image(img_log, "6b – Transformacja logarytmiczna (c=0.5)")

# c) Zmiana kontrastu (sigmoid)
img_contrast = contrast_transform(img_c, m=0.45, e=8)
show_image(img_contrast, "6c – Zmiana kontrastu (m=0.45, e=8)")

# d) Korekcja gamma
img_gamma = gamma_correction(img_d, c=1.0, gamma=0.4)
show_image(img_gamma, "6d – Korekcja gamma (gamma=0.4)")
