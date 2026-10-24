from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import filemanagement as fm

selfie_path = Path("images/selfie.jpg")
soda_path = Path("images/100.tif")

selfie_img = Image.open(selfie_path)
soda_img = Image.open(soda_path)


def hori_blur(img: Image.Image) -> Image.Image:
    img_arr = np.asarray(img.convert("L"), dtype=np.float32)
    kernel = (1 / 7) * np.ones((1, 7), dtype=np.float32)
    blurred = cv.filter2D(img_arr, -1, kernel)
    return Image.fromarray(blurred.round().astype(np.uint8))


selfie_blur = hori_blur(selfie_img)
selfie_blur.save(fm.output_path(selfie_path, "blur"))
plt.imshow(selfie_blur, cmap="gray")
plt.show()

soda_blur = hori_blur(soda_img)
soda_blur.save(fm.output_dir.joinpath("soda_blur.png"))
plt.imshow(soda_blur, cmap="gray")
plt.show()
