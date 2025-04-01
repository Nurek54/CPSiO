from skimage import io
import numpy as np
import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Function to load and display grayscale image
def load_and_display_image(path):
    image = io.imread(path, as_gray=True)
    plt.imshow(image, cmap='gray')
    plt.title("Obraz w skali szarości")
    plt.axis('off')
    plt.show()
    return image


# Function to plot horizontal or vertical intensity profile
def plot_intensity_profile(image, row=None, column=None):
    plt.figure()
    if row is not None:
        plt.plot(image[row, :], label=f'Wiersz {row}')
        plt.title(f'Profil jasności – wiersz {row}')
    elif column is not None:
        plt.plot(image[:, column], label=f'Kolumna {column}')
        plt.title(f'Profil jasności – kolumna {column}')
    else:
        print("Podaj row=... lub column=...")
        return
    plt.xlabel('Pozycja piksela')
    plt.ylabel('Poziom jasności')
    plt.grid()
    plt.legend()
    plt.show()


# Function to crop selected region and save to file
def crop_and_save_image(image, top, left, height, width, save_path):
    cropped = image[top:top + height, left:left + width]
    io.imsave(save_path, (cropped * 255).astype(np.uint8))  # convert to uint8 before saving
    plt.imshow(cropped, cmap='gray')
    plt.title("Wycięty fragment obrazu")
    plt.axis('off')
    plt.show()
    print(f"Zapisano fragment obrazu do: {os.path.abspath(save_path)}")


# Example usage:
if __name__ == "__main__":
    # Replace this path with your image path
    image_path = r"chest-xray.tif"

    # Load and show full image
    img = load_and_display_image(image_path)

    # Show intensity profile for row 100
    plot_intensity_profile(img, row=100)

    # Crop and save a 100x100 region starting at (50, 50)
    crop_and_save_image(img, top=50, left=50, height=100, width=100,
                        save_path=r"cropped_xray.tif")
