import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image_and_hist(title, image, position):
    plt.subplot(2, 4, position)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

    plt.subplot(2, 4, position + 4)
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title('Histogram')

# Wczytaj obraz w odcieniach szarości
original_img = cv2.imread('Lista2/chest-xray.tif', cv2.IMREAD_GRAYSCALE)

# Symuluj zbyt ciemny i zbyt jasny obraz
dark_img = np.clip(original_img * 0.4, 0, 255).astype(np.uint8)
bright_img = np.clip(original_img * 1.6, 0, 255).astype(np.uint8)

# Wyrównywanie histogramu
dark_eq = cv2.equalizeHist(dark_img)
bright_eq = cv2.equalizeHist(bright_img)

# Wyświetl wyniki
plt.figure(figsize=(16, 8))

show_image_and_hist('Ciemny obraz', dark_img, 1)
show_image_and_hist('Ciemny po wyrównaniu', dark_eq, 2)
show_image_and_hist('Jasny obraz', bright_img, 3)
show_image_and_hist('Jasny po wyrównaniu', bright_eq, 4)

plt.tight_layout()
plt.show()
