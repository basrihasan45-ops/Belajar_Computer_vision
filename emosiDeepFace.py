import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

img=cv2.imread('emosi2.jpg')
plt.imshow(img[:, :, : : -1])

plt.show()

result = DeepFace.analyze(img, actions=['emotion'])
print(result)