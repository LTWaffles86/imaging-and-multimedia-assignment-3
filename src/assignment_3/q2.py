import file_management

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ireland_path = Path("images/ireland-03gray.tif")


ireland_img = Image.open(ireland_path).convert("L")


# clips to the maximum of the data type of the first array
def saturating_add(*arrays: np.ndarray) -> np.ndarray:
    dtype = arrays[0].dtype
    info = np.iinfo(dtype)

    result = arrays[0].astype(np.int64)
    for a in arrays[1:]:
        result += a.astype(np.int64)

    return np.clip(result, info.min, info.max).astype(dtype)


def highlight_edges(
    image: Image.Image, kernel_x: np.ndarray, kernel_y: np.ndarray, scale: float
) -> Image.Image:
    img_arr = np.asarray(image)
    edges_x = cv.filter2D(img_arr, cv.CV_32F, kernel_x)
    edges_y = cv.filter2D(img_arr, cv.CV_32F, kernel_y)
    magnitude = np.sqrt(edges_x * edges_x + edges_y * edges_y)
    highlighted = saturating_add(
        img_arr,
        scale * magnitude,  # scale magnitude so it clips less often
    )
    return Image.fromarray(highlighted)


def highlight_edges_color(
    image: Image.Image, kernel_x: np.ndarray, kernel_y: np.ndarray, scale: float
) -> Image.Image:
    img_arr = np.asarray(image)
    grayscale_arr = img_arr.mean(-1, dtype=np.float32)
    edges_x = cv.filter2D(grayscale_arr, cv.CV_32F, kernel_x)
    edges_y = cv.filter2D(grayscale_arr, cv.CV_32F, kernel_y)
    magnitude = np.sqrt(edges_x * edges_x + edges_y * edges_y)
    highlighted = saturating_add(
        img_arr,
        scale * magnitude[:, :, None],
    )
    return Image.fromarray(highlighted)


def darken_edges_color(
    image: Image.Image, kernel_x: np.ndarray, kernel_y: np.ndarray, scale: float
) -> Image.Image:
    return highlight_edges_color(image, kernel_x, kernel_y, -scale)


plt.imshow(ireland_img, cmap="gray")
plt.show()

prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.int8)
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.int8)
ireland_prewitt = highlight_edges(ireland_img, prewitt_x, prewitt_y, 0.5)

ireland_prewitt.save(output_path(ireland_path, "prewitt"))
plt.imshow(ireland_prewitt, cmap="gray")
plt.show()


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int8)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int8)
ireland_sobel = highlight_edges(ireland_img, sobel_x, sobel_y, 0.5)
ireland_sobel.save(output_path(ireland_path, "sobel"))
plt.imshow(ireland_sobel, cmap="gray")
plt.show()

amsterdam_path = Path("images/Amsterdam.jpg")
amsterdam_img = Image.open(amsterdam_path)

plt.imshow(amsterdam_img)
plt.show()

amsterdam_prewitt = highlight_edges_color(amsterdam_img, prewitt_x, prewitt_y, 0.5)
amsterdam_prewitt.save(output_path(amsterdam_path, "prewitt"))
plt.imshow(amsterdam_prewitt)
plt.show()

amsterdam_sobel = highlight_edges_color(amsterdam_img, sobel_x, sobel_y, 0.5)
amsterdam_sobel.save(output_path(amsterdam_path, "sobel"))
plt.imshow(amsterdam_sobel)
plt.show()


amsterdam_prewitt_dark = darken_edges_color(amsterdam_img, prewitt_x, prewitt_y, 0.5)
amsterdam_prewitt_dark.save(output_path(amsterdam_path, "prewitt_dark"))
plt.imshow(amsterdam_prewitt_dark)
plt.show()

amsterdam_sobel_dark = darken_edges_color(amsterdam_img, sobel_x, sobel_y, 0.5)
amsterdam_sobel_dark.save(output_path(amsterdam_path, "sobel_dark"))
plt.imshow(amsterdam_sobel_dark)
plt.show()
