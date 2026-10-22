from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

selfie_path = Path("images/selfie.jpg")
output_path = selfie_path.parent.joinpath("output")
if not output_path.exists():
    output_path.mkdir()

x = 406
y = 155
w = 974
h = 1351

img = Image.open(selfie_path)
img_arr = np.asarray(img)
roi = img_arr[y : y + h, x : x + w]

for k in [15, 25, 50]:
    kernel = np.ones((k, k), np.float32) / (k * k)
    blurred = cv.filter2D(roi, -1, kernel)
    out = img_arr.copy()
    out[y : y + h, x : x + w] = blurred

    blurred_img = Image.fromarray(out)
    blurred_img.save(output_path.joinpath(f"blurred_selfie_{k}.jpg"))

    plt.imshow(blurred_img)
    plt.show()

    small = cv.resize(
        roi, (roi.shape[1] // k, roi.shape[0] // k), interpolation=cv.INTER_AREA
    )
    pixelated = cv.resize(
        small, (roi.shape[1], roi.shape[0]), interpolation=cv.INTER_NEAREST
    )

    out[y : y + h, x : x + w] = pixelated
    pixelated_img = Image.fromarray(out)
    pixelated_img.save(output_path.joinpath(f"pixelated_selfie_{k}.jpg"))

    plt.imshow(pixelated_img)
    plt.show()
