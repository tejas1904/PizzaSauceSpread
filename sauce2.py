from matplotlib import pyplot as plt
import cv2
import skimage.io
import numpy as np
from skimage import data
from skimage.color import rgb2gray, rgb2hsv
from skimage import filters
from skimage.filters import threshold_otsu
from skimage import data, io
from skimage import morphology
from skimage.segmentation import flood, flood_fill
from matplotlib import pyplot as plt

def quantify_colour(final_img):
    H = final_img[:, :, 0].ravel()
    s = final_img[:, :, 1].ravel()
    bins = np.arange(0, 1.001, (1 / 16))  # 1.001 to include 1 in the bins as its 0 to n-1
    bins[-1] = 1.001  # so that digitize function digitzes the 1 also and not until 0.9999 making 0-1 as 0-1.00001

    H_inds = np.digitize(H, bins)
    S_inds = np.digitize(s, bins)
    # print(bins)
    code_block = {}  # DICT
    n = 0
    for i in range(1, 17):
        for j in range(1, 17):
            s = (i, j)
            code_block[s] = n
            n = n + 1

    block_numbers = []

    for i in zip(H_inds, S_inds):
        block_numbers.append(code_block[i])

    return block_numbers
    # plt.hist(block_numbers)
    # plt.show()

def segmentation(im):
    im_gray = rgb2gray(im)
    im_gray = filters.gaussian(im_gray, 2)
    im_sobel = filters.sobel(im_gray)
    im_sobel = filters.sobel(im_gray)
    thresh_img = (im_sobel > threshold_otsu(im_sobel)) * 1.0
    dialated_img = morphology.dilation(thresh_img)
    flood_img = flood_fill(dialated_img, (0, 0), 1.0)
    neg_flood_img = (flood_img - 1) * -1
    mask = np.logical_or(thresh_img, neg_flood_img) * 1.0
    eroded_mask = morphology.dilation(mask)
    eroded_maski = eroded_mask.astype(int)

    final = np.zeros_like(im)
    for i in range(3):
        final[:, :, i] = im[:, :, i] * eroded_maski

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
    axes[0].imshow(im)
    axes[1].imshow(im_sobel, cmap='gray')
    axes[2].imshow(final)
    plt.show()

    return final

#im=image = skimage.io.imread(fname="images/pizza1.jpg")
im = image = skimage.io.imread(fname="pizzaSause/dup2.png")
#im=image = skimage.io.imread(fname="ImageSet/01.png")
final=segmentation(im)
final = rgb2hsv(final)
quantify=quantify_colour(final)

value,counts = np.unique(quantify,return_counts=True) #thats it feed this into svm thats all
for i in zip(value,counts):
    print(i)

plt.hist(quantify)
plt.show()








