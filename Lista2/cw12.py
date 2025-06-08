import matplotlib
matplotlib.use('TkAgg')

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte, exposure, filters, morphology, metrics

def save(i, n, d):
    io.imsave(os.path.join(d, n + '.tif'), img_as_ubyte(i), check_contrast=False)

def show(i, t):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(i, cmap='gray')
    ax.set_title(t)
    ax.axis('off')
    plt.show()
    plt.close(fig)

def denoise(x):
    return filters.gaussian(filters.median(x, morphology.disk(3)), sigma=1)

def clahe(x):
    return exposure.equalize_adapthist(x, clip_limit=0.03)

def sharpen(x):
    r, a, k = 2, 1.5, 1.2
    u = filters.unsharp_mask(x, radius=r, amount=a)
    g = filters.gaussian(x, sigma=r)
    h = np.clip(x + k * (x - g), 0, 1)
    return np.clip(0.5 * u + 0.5 * h, 0, 1)

def smooth(x):
    return filters.gaussian(x, sigma=0.5)

def morph(x):
    return x * morphology.remove_small_objects(x > 0.15, 500)

def main(src='bonescan.tif', dst='wyniki'):
    if not os.path.exists(src):
        print('Brak pliku:', src)
        sys.exit(1)
    os.makedirs(dst, exist_ok=True)

    o = img_as_float(io.imread(src, as_gray=True)); save(o, '01_oryginal', dst); show(o, '01 Oryginal')

    d = denoise(o); save(d, '02_denoised', dst); show(d, '02 Denoised')

    c = clahe(d); save(c, '03_contrast', dst); show(c, '03 CLAHE')

    s = sharpen(c); save(s, '04_sharpened', dst); show(s, '04 Sharpen')

    f = morph(smooth(s)); save(f, '05_final', dst); show(f, '05 Final')

    psnr = metrics.peak_signal_noise_ratio(o, f, data_range=1)
    ssim = metrics.structural_similarity(o, f, data_range=1)
    print('PSNR', psnr, 'SSIM', ssim)

if __name__ == '__main__':
    main()
