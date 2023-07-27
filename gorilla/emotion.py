import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

img = cv2.imread(r'C:\Users\RINA1\Desktop\gorilla\h1.jpg')

plt.imshow(img[:,:,::-1])

plt.show()

result = DeepFace.analyze(img,actions=['emotion'])

print(result)