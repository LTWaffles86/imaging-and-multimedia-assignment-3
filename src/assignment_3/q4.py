from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import filemanagement as fm

paths = [Path(f"images/{x}.tif") for x in range(100, 111)]
imgs = [Image.open(path).convert("L") for path in paths]


def alpha_blend(alpha: float, images: list[Image.Image]) -> Image.Image:
    frames = [np.asarray(img, dtype=np.float32) for img in images]
    B = frames[0]
    for I in frames[1:]:
        B = alpha * B + (1 - alpha) * I
    return Image.fromarray(B.round().astype(np.uint8))


for alpha in [0.1, 0.2, 0.5, 0.8]:
    blended = alpha_blend(alpha, imgs)
    blended.save(fm.output_dir.joinpath(f"blend_{alpha}.png"))
    plt.imshow(blended, cmap="gray", vmin=0, vmax=255)
    plt.show()
