import os
import uuid
import numpy as np

import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt

from skimage import io, img_as_float, filters
from scipy.ndimage import gaussian_filter, laplace

def rescale01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    return (arr - arr.min()) / (np.ptp(arr) + 1e-12)

def safe_show() -> None:
    try:
        plt.show()
    except AttributeError as e:
        fname = f"figure_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(fname, dpi=150)
        plt.close()

def show_edges(original: np.ndarray, edges_dict: dict[str, np.ndarray], *, title: str = "Sobel") -> None:
    n = len(edges_dict) + 1
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))
    axs[0].imshow(original, cmap="gray"); axs[0].set_title("Oryginał"); axs[0].axis("off")
    for i, (label, arr) in enumerate(edges_dict.items(), start=1):
        axs[i].imshow(rescale01(arr), cmap="gray")
        axs[i].set_title(label)
        axs[i].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    safe_show()

def show_sharpening(original: np.ndarray, mask: np.ndarray, sharpened: np.ndarray, *, title: str = "") -> None:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original, cmap="gray");  axs[0].set_title("Oryginał");               axs[0].axis("off")
    axs[1].imshow(rescale01(mask), cmap="gray"); axs[1].set_title("Maska / high-pass"); axs[1].axis("off")
    axs[2].imshow(sharpened, cmap="gray");  axs[2].set_title("Po wyostrzeniu");        axs[2].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    safe_show()

def sobel_edges(image: np.ndarray) -> dict[str, np.ndarray]:
    s_h = filters.sobel_h(image)
    s_v = filters.sobel_v(image)
    magnitude = np.hypot(s_h, s_v)
    diag45  = (s_h + s_v) / np.sqrt(2)
    diag135 = (s_h - s_v) / np.sqrt(2)
    return {
        "Poziome": s_h,
        "Pionowe": s_v,
        "Ukośne 45°": diag45,
        "Ukośne 135°": diag135,
        "Magnituda": magnitude,
    }

def laplace_highpass(image: np.ndarray) -> np.ndarray:
    return laplace(image)

def unsharp_mask(image: np.ndarray, *, sigma: float = 1.0):
    blurred = gaussian_filter(image, sigma=sigma)
    mask = image - blurred
    sharpened = np.clip(image + mask, 0, 1)
    return mask, sharpened

def highboost(image: np.ndarray, *, sigma: float = 1.0, k: float = 3.0):
    blurred = gaussian_filter(image, sigma=sigma)
    mask = image - blurred
    sharpened = np.clip(image + k * mask, 0, 1)
    return mask, sharpened

def process() -> None:
    sobel_files = ["circuitmask.tif", "testpat1.png"]
    for f in sobel_files:
        if not os.path.exists(f):
            print(f"❌ Brak pliku: {f}")
            continue
        img = img_as_float(io.imread(f, as_gray=True))
        edges = sobel_edges(img)
        show_edges(img, edges, title=f"Sobel – {os.path.basename(f)}")

    laplace_file = "blurry-moon.tif"
    if os.path.exists(laplace_file):
        img = img_as_float(io.imread(laplace_file, as_gray=True))
        hp = laplace_highpass(img)
        sharpened = np.clip(img - hp, 0, 1)
        show_sharpening(img, hp, sharpened, title="Laplace – wyostrzanie")
    else:
        print(f"❌ Brak pliku: {laplace_file}")

    text_file = "text-dipxe-blurred.tif"
    if os.path.exists(text_file):
        img = img_as_float(io.imread(text_file, as_gray=True))
        mask_u, sharp_u = unsharp_mask(img, sigma=1.0)
        show_sharpening(img, mask_u, sharp_u, title="Unsharp masking (σ=1.0)")

        mask_hb, sharp_hb = highboost(img, sigma=1.0, k=3.0)
        show_sharpening(img, mask_hb, sharp_hb, title="High-boost (k=3.0, σ=1.0)")
    else:
        print(f"❌ Brak pliku: {text_file}")

if __name__ == "__main__":
    process()
