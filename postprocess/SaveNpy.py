import tensorflow as tf
import numpy as np
import glob  #查找符合特定规则的文件路径名
from PIL import Image
 
Datapath = "./img_processed3/*.png"
imgs = np.zeros((404, 1024, 1024))
i = 0
for i in range(404):
	# 打开图像并转化为数字矩阵(240x240x3)
	img = np.array(Image.open("./img_processed3/"+str(i)+".png"))
	imgs[i] = img
np.save('a.npy',imgs)