from skimage.io import imread
import matplotlib.pylab as plt
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
from PIL import Image

im=Image.open('cropped_good_spread/1.jpg')
im=np.array(im)
print(im.shape)


#im=imread("cropped_good_spread/1.jpg")
im=rescale(im, 0.15, anti_aliasing=False,multichannel=True)
print(np.shape(im))
fig,axes= plt.subplots(ncols=4, nrows=1)
axes[0].imshow(im)
axes[1].imshow(im[:,:,0],cmap='gray')
axes[2].imshow(im[:,:,1],cmap='gray')
axes[3].imshow(im[:,:,2],cmap='gray')
plt.show()

plt.hist(np.ravel(im[:,:,0])*256,bins=256,color='r',alpha=0.5)
plt.hist(np.ravel(im[:,:,1])*256,bins=256,color='g',alpha=0.5)
plt.hist(np.ravel(im[:,:,2])*256,bins=256,color='b',alpha=0.5)
plt.show()
print(np.ravel(im[:,:,0]))



