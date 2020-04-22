import numpy as np
from PIL import Image

def apply_filter(image, kernel):

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    h = kernel_h // 2
    w = kernel_w // 2

    image = np.pad(image, pad_width=((h,h),(w,w)), mode='edge').astype(np.float32)
    filtered_img = np.zeros(image.shape)

    for i in range(h, image.shape[0]-h):
        for j in range(w, image.shape[1]-w):
            x = image[i-h : i-h+kernel_h , j-w : j-w+kernel_w]
            x = x.flatten() * kernel.flatten()
            filtered_img[i][j] = x.sum()

    return filtered_img[h:-h,w:-w]

def get_gaussian_filter(sigma,size):

    s = size//2
    g_filter = np.zeros((size,size), np.float32)

    for x in range(-s, s+1):
        for y in range(-s, s+1):
            num = np.exp(-(x**2 + y**2)/(2*sigma**2))
            denom = 2*np.pi*(sigma**2)
            g_filter[x+s,y+s] = num/denom

    return g_filter

def gaussian_blur(image, sigma, k_size):

    image = np.asarray(Image.open(image))
    g_filter = get_gaussian_filter(sigma,k_size)
    blur_img = np.zeros_like(image, dtype=np.float32)

    for c in range(3):
        blur_img[:, :, c] = apply_filter(image[:, :, c], g_filter)
    return Image.fromarray(blur_img.astype(np.uint8))
