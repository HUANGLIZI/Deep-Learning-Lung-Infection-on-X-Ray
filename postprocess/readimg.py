import numpy as np
import cv2
import matplotlib.pyplot as plt
image = np.load("a.npy")
print(image.shape)
for i in range(3,5):
    plt.imshow(image[i,:,:])
    cv2.imwrite(str(i)+".png",image[i,:,:])
    #plt.show()