import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img=cv.cvtColor(cv.imread('objekKucing.jpg'), cv.COLOR_BGR2RGB)
blur=cv.GaussianBlur(img,(5,5),0)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur), plt.title('Blurred')
plt.show()