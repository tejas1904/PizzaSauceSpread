from matplotlib import pyplot as plt
import cv2
import skimage.io
import numpy as np
from skimage import data
from skimage.color import rgb2gray, rgb2hsv
from skimage import filters
from skimage.filters import threshold_otsu,unsharp_mask
from skimage import data, io
from skimage import morphology
from skimage.segmentation import flood, flood_fill
from matplotlib import pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.morphology import disk
from skimage import exposure
from skimage import feature

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
    #im_gray = unsharp_mask(im_gray, radius=20, amount=2)
    im_gray = filters.gaussian(im_gray, 2)
    im_sobel = filters.sobel(im_gray)
    thresh_img = (im_sobel > filters.threshold_li(im_sobel)) * 1.0
    dialated_img = morphology.dilation(thresh_img,selem=disk(2))
    flood_img = flood_fill(dialated_img, (0, 0), 1.0)
    neg_flood_img = (flood_img - 1) * -1

    mask = np.logical_or(thresh_img, neg_flood_img) * 1.0
    eroded_mask = morphology.dilation(mask,selem=disk(2))
    eroded_mask = morphology.erosion(eroded_mask, selem=disk(8))

    eroded_maski = eroded_mask.astype(int)

    final = np.zeros_like(im)
    for i in range(3):
        final[:, :, i] = im[:, :, i] * eroded_maski

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(11, 11),)
    fig.tight_layout()
    axes[0][0].imshow(im)
    axes[0][1].imshow(im_gray, cmap='gray')
    axes[0][2].imshow(im_sobel,cmap='gray')
    axes[0][3].imshow(thresh_img,cmap='gray')
    axes[0][4].imshow(dialated_img, cmap='gray')
    axes[1][0].imshow(flood_img, cmap='gray')
    axes[1][1].imshow(neg_flood_img, cmap='gray')
    axes[1][2].imshow(mask, cmap='gray')
    axes[1][3].imshow(eroded_maski,cmap='gray')
    axes[1][4].imshow(final, cmap='gray')
    # for ii in range(2):
    #     for j in range(5):
    #         axes[ii,j].axis('off')



    plt.show()

    return final

def histogram(filename):
        im = skimage.io.imread(filename)
        im= rescale(im, 0.2, anti_aliasing=True,multichannel=True)
        final=segmentation(im)
        final = rgb2hsv(final)
        quantify=quantify_colour(final)

        value,counts = np.unique(quantify,return_counts=True) #thats it feed this into svm thats all
        x = []
        for i in range(len(value)):
            x.append((value[i], counts[i]))

        for i in range(0, 256):
            if i not in value:
                x.append((i, 0))

        #plt.hist(quantify)
        #plt.show()

        return sorted(x, key=lambda a:a[0])
for i in range(1,25):
    im=skimage.io.imread(f"cropped_good_spread/{i}.jpg")
    im=rescale(im, 0.25, anti_aliasing=False,multichannel=True)
    segmentation(im)