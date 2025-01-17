import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread(r'"C:/Users/priyy/Desktop/Dora.jpeg"')

b, g, r= cv2.split(img)

rgb_img = cv2.merge([r, g, b])

gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2. THRESH_OTSU)

kernel =np.ones((2, 2), np.uint8)

closing =cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

sure_bg= cv2.dilate(closing, kernel, iterations=3)

dist_transform = cv2.distanceTransform (closing, cv2.DIST_L2, 3)

ret, sure_fg = cv2. threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)

sure_fg= np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
markers =markers + 1
markers [unknown == 255] = 0
markers =cv2.watershed(img, markers)
img[markers == -1] =[255, 0, 0]

plt.subplot(211)

plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))

plt.title('Input Image')

plt.xticks([]), plt.yticks([])

plt.subplot(212)

plt.imshow(thresh, cmap='gray')

plt.title("Otsu's Binary Threshold")

plt.xticks([]), plt.yticks([])

plt.imsave(r'C:/Users/priyy/Desktop/Dora.jpeg', thresh, cmap='gray')

plt.tight_layout()

plt.show()
