import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img=cv.imread('objekKucing.jpg',0)
edges=cv.Canny(img,100,200)

plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title('original image')
plt.xticks([]),plt.yticks([])

plt.subplot(122)
plt.imshow(edges,cmap='gray')
plt.title('edge image')
plt.xticks([]),plt.yticks([])
plt.show