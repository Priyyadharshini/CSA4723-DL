import numpy as np

import matplotlib.pyplot as plt
from skimage import data, color
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.measure import label
image =color.rgb2gray(data.astronaut())
elevation_map= sobel (image)
markers= np.zeros_like(image, dtype=np.int32)
markers [image < 0.3]= 1

markers [image > 0.7] = 2

segmentation =watershed(elevation_map, markers, mask=image)

labels= label(segmentation)

fig, axs= plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(image, cmap='gray')

axs[0].set_title('Original Image')

axs[0].axis('off')

axs[1].imshow(color.label2rgb(labels, image=image))

axs[1].set_title('Segmented Image')

axs[1].axis('off')

plt.show()
