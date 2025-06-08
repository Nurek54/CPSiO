import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_and_display_image(path):
    img = Image.open(path)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

def plot_intensity(path, row, column):
    img_gray = Image.open(path).convert("L")
    arr = np.array(img_gray)

    plt.figure(figsize=(10, 10))
    title = "Image in Grayscale & Gray Intensity Plots"
    plt.suptitle(title)

    # increase the vertical gap between subplots:
    plt.subplots_adjust(hspace=0.7)

    plt.subplot(3, 1, 1)
    plt.imshow(img_gray, cmap="gray")
    plt.title("Image in Grayscale")
    plt.axis("off")

    plt.subplot(3, 1, 2)
    plt.plot(arr[row])
    plt.title(f"Horizontal Grayscale in row {row}")

    plt.subplot(3, 1, 3)
    plt.plot(arr[:, column])
    plt.title(f"Vertical Grayscale in column {column}")

    plt.show()


def crop_image(path, x1, y1, x2, y2):
    img_cropped = Image.open(path).crop((x1, y1, x2, y2))
    plt.imshow(img_cropped)
    plt.title("Cropped Image")
    plt.axis("off")
    plt.show()



load_and_display_image("Lista2/aerial_view.tif")
plot_intensity("Lista2/aerial_view.tif", 50, 250)
crop_image("Lista2/aerial_view.tif", 50, 50, 250, 250)