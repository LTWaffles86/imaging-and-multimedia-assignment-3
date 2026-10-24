from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

import filemanagement as fm

soccer1_path = Path("images/soccer1.bmp")
soccer2_path = Path("images/soccer2.bmp")
soccer1_img = Image.open(soccer1_path)
soccer2_img = Image.open(soccer2_path)


def difference_image(img1: Image.Image, img2: Image.Image) -> Image.Image:
    arr1 = np.asarray(img1).astype(np.int16)
    arr2 = np.asarray(img2).astype(np.int16)
    d = np.abs(arr1 - arr2).astype(np.uint8)
    return Image.fromarray(d)


diff_img = difference_image(soccer1_img, soccer2_img)
diff_img_arr = np.array(diff_img)
plt.hist(diff_img_arr.ravel(), bins=256)
plt.savefig(fm.output_dir.joinpath("diff_img_hist.png"))
plt.show()

diff_img_stretched = ImageOps.autocontrast(diff_img, cutoff=1)
diff_img_stretched.save(fm.output_dir.joinpath("diff_img_stretched.png"))
diff_img_stretched_arr = np.asarray(diff_img_stretched).ravel()
plt.hist(diff_img_stretched_arr, bins=256)
plt.savefig(fm.output_dir.joinpath("diff_img_stretched_hist.png"))
plt.show()
plt.imshow(diff_img_stretched, cmap="gray")
plt.show()

ps = [2.0, 1.0, 0.1, 0.01]
percentiles = np.percentile(diff_img_arr.ravel(), [100.0 - p for p in ps])

for p, percentile in zip(ps, percentiles):
    binary_diff_img = Image.fromarray(diff_img_arr > percentile)
    binary_diff_img.save(fm.output_dir.joinpath(f"soccer_binary_{p}.png"))
    plt.imshow(binary_diff_img)
    plt.show()

best_percentile = np.percentile(diff_img_arr.ravel(), 99.0)
mask_arr = diff_img_arr > best_percentile
masked_img = Image.fromarray(np.array(soccer2_img) * mask_arr)
masked_img.save(fm.output_dir.joinpath("soccer2_masked.png"))
plt.imshow(masked_img, cmap="gray")
plt.show()


security1_path = Path("images/1015.jpg")
security2_path = Path("images/1020.jpg")
security3_path = Path("images/1025.jpg")

security1_img = Image.open(security1_path).convert("L")
security2_img = Image.open(security2_path).convert("L")
security3_img = Image.open(security3_path).convert("L")

security_diff_12 = difference_image(security1_img, security2_img)
security_diff_23 = difference_image(security2_img, security3_img)

security_diff_12_stretched = ImageOps.autocontrast(security_diff_12, cutoff=1)
security_diff_23_stretched = ImageOps.autocontrast(security_diff_23, cutoff=1)

plt.imshow(security_diff_12_stretched, cmap="gray")
plt.show()
plt.imshow(security_diff_23_stretched, cmap="gray")
plt.show()

security_diff_12_arr = np.asarray(security_diff_12)
security_diff_23_arr = np.asarray(security_diff_23)

percentile_12 = np.percentile(security_diff_12_arr, 99.0)
percentile_23 = np.percentile(security_diff_23_arr, 99.0)

masked_img2 = Image.fromarray(
    np.asarray(security2_img) * (security_diff_12_arr > percentile_12)
)

masked_img2.save(fm.output_path(security2_path, "masked"))

masked_img3 = Image.fromarray(
    np.asarray(security3_img) * (security_diff_23_arr > percentile_23)
)
masked_img3.save(fm.output_path(security3_path, "masked"))
