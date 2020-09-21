import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove(j):
    str1="img/"+str(j)+".png"
    img = cv2.imread(str1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_temp = gray.copy()  
    contours, hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
    plt.figure("original image with contours"), plt.imshow(img, cmap = "gray")
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    area[max_idx] = np.min(area)
    second_idx=np.argmax(area)
    #print(len(contours))
    #print(max_idx)
    #print(second_idx)
    if max_idx!=second_idx:
        for i in range(len(contours)):
            if i!=max_idx:
                if i!=second_idx:
                    cv2.fillConvexPoly(gray, contours[i], 0)
#cv2.fillConvexPoly(gray, contours[second_idx], 0)
    plt.figure("remove small connect com"),plt.imshow(gray, cmap = "gray")
    cv2.imwrite(str(j)+".png",gray)
#plt.show()
for j in range(404):
    remove(j)