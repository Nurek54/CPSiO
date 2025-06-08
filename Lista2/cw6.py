import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def constant():
    images = ["Lista2/chest-xray.tif", "Lista2/pollen-dark.tif", "Lista2/spectrum.tif"]

    constants = [0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0]

    for image in images:
        img_gray = Image.open(image).convert("L")
        arr = np.array(img_gray, dtype=float) # data converted to float, to icrease accuracy of multiplying

        plt.figure(figsize=(10,10))
        plt.suptitle(f"Stała: {image}")

        for i, c in enumerate(constants):
            img_multiplied = np.clip(arr * c, 0, 255).astype(np.uint8) # data clipped to stay in 0-255 values and uint8
            plt.subplot(3, 3, i+1)
            plt.imshow(img_multiplied, cmap="gray", vmin=0, vmax=255)
            plt.axis("off")
            plt.title(f"c = {c}")
        
        plt.show()

def log():
    images = ["Lista2/spectrum.tif"]

    constants = [0.25, 0.5, 1, 5, 10, 25, 50, 100, 150]

    for image in images:
        img_gray = Image.open(image).convert("L")
        arr = np.array(img_gray, dtype=float) # data converted to float, to icrease accuracy of multiplying

        plt.figure(figsize=(10,10))
        plt.suptitle(f"Logarytmiczna: {image}")

        for i, c in enumerate(constants):
            img_multiplied = np.clip(c * np.log(1 + arr), 0, 255).astype(np.uint8) # data clipped to stay in 0-255 values and uint8
            plt.subplot(3, 3, i+1)
            plt.imshow((img_multiplied / img_multiplied.max()) * 255, cmap="gray", vmin=0, vmax=255) # normalized to fill all values 0-255
            plt.axis("off")
            plt.title(f"c = {c}")
        
        plt.show()

def dynamics_and_grayscale():
    images = ["Lista2/chest-xray.tif", "Lista2/einstein-low-contrast.tif", "Lista2/pollen-lowcontrast.tif"]

    ms = [0.45, 1.0, 1.5]
    es = [4, 8, 12]

    for image in images:
        img_gray = Image.open(image).convert("L")
        arr = np.array(img_gray, dtype=float)
        
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
        
        plt.figure(figsize=(10,10))
        plt.suptitle(f"Transformacja sigmoidalna: {image}")

        counter = 1

        for m in ms:
            for e in es:
                with np.errstate(divide='ignore', invalid='ignore'):
                    transformed = 1 / (1 + (m / np.where(arr_norm == 0, np.inf, arr_norm)) ** e)
                    transformed[arr_norm == 0] = 0
                
                result = (255 * (transformed - transformed.min()) / 
                        (transformed.max() - transformed.min())).astype(np.uint8)
                
                plt.subplot(3, 3, counter)
                plt.imshow(result, cmap="gray", vmin=0, vmax=255)
                plt.axis("off")
                plt.title(f"m = {m}, e = {e}")
                counter += 1
        
        plt.show()

def gamma_correction():
    images = ["Lista2/aerial_view.tif"]

    constants = [0.5, 1.0, 1.5]
    gammas = [0.5, 1, 1.5]

    for image in images:
        img_gray = Image.open(image).convert("L")
        arr = np.array(img_gray, dtype=float)
        
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
        
        plt.figure(figsize=(10,10))
        plt.suptitle(f"Transformacja sigmoidalna: {image}")

        counter = 1

        for c in constants:
            for g in gammas:
                transformed = c * (arr_norm ** g)
                result = (255 * (transformed - transformed.min()) / 
                        (transformed.max() - transformed.min())).astype(np.uint8)
                
                plt.subplot(3, 3, counter)
                plt.imshow(result, cmap="gray", vmin=0, vmax=255)
                plt.axis("off")
                plt.title(f"c = {c}, gamma = {g}")
                counter += 1
        
        plt.show()
        
# constant()
# log()
dynamics_and_grayscale()
# gamma_correction()