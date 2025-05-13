import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj obraz
img = cv2.imread('Lista2/chest-xray.tif', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Nie udało się wczytać obrazu.")

# Parametry masek do testu
mask_sizes = [3, 8, 15]

plt.figure(figsize=(12, 10))

for i, size in enumerate(mask_sizes):
    # a) CLAHE – lokalne wyrównywanie histogramu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(size, size))
    clahe_img = clahe.apply(img)

    # b) Lokalna poprawa jakości (lokalne rozciąganie histogramu)
    local_mean = cv2.blur(img.astype(np.float32), (size, size))
    local_sqmean = cv2.blur(np.square(img.astype(np.float32)), (size, size))
    local_std = np.sqrt(local_sqmean - local_mean ** 2)

    # Poprawiony obraz (na podstawie lokalnych statystyk)
    enhanced_img = np.clip((img - local_mean) / (local_std + 1e-5) * 32 + 128, 0, 255).astype(np.uint8)

    # Wyświetl wyniki
    plt.subplot(len(mask_sizes), 3, i*3 + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Oryginał (mask={size})')
    plt.axis('off')

    plt.subplot(len(mask_sizes), 3, i*3 + 2)
    plt.imshow(clahe_img, cmap='gray')
    plt.title(f'CLAHE (mask={size}x{size})')
    plt.axis('off')

    plt.subplot(len(mask_sizes), 3, i*3 + 3)
    plt.imshow(enhanced_img, cmap='gray')
    plt.title(f'Lokalna statystyka (mask={size}x{size})')
    plt.axis('off')

plt.tight_layout()
plt.show()
