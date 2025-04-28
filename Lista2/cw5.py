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
    img_gray = Image.open(path).convert("L") # L parameter provides conversion of data to gray scale
    arr = np.array(img_gray)

    plt.figure(figsize=(10, 10))
    title = "Image in Grayscale & Gray Intensity Plots"
    plt.suptitle(title)

    plt.subplot(3, 1, 1)
    plt.imshow(img_gray, cmap="gray")
    plt.title("Image in Grayscale")
    plt.axis("off")

    plt.subplot(3, 1, 2)
    plt.plot(arr[row])
    plt.title("Horizontal Grayscale in row")

    plt.subplot(3, 1, 3)
    plt.plot(arr[:, column])
    plt.title("Vertical Grayscale in column")

    plt.show()

def crop_image(path, x1, y1, x2, y2):
    img_cropped = Image.open(path).crop((x1, y1, x2, y2))
    plt.imshow(img_cropped)
    plt.title("Cropped Image")
    plt.axis("off")
    plt.show()



load_and_display_image("Lista2/aerial_view.tif")
plot_intensity("Lista2/aerial_view.tif", 200, 200)
crop_image("Lista2/aerial_view.tif", 0, 0, 180, 180)