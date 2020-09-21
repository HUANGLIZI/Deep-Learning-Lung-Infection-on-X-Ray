import os
import os.path
from PIL import Image
'''
filein: 输入图片
fileout: 输出图片
width: 输出图片宽度
height:输出图片高度
type:输出图片类型（png, gif, jpeg...）
'''
def ResizeImage(filein, fileout, width, height):
  img = Image.open(filein)
  out = img.resize((width, height))
  #resize image with high-quality
  out.save(fileout)
def Resize(i):
  filein = r"./img_processed/"+str(i)+".png"
  fileout = r"./img_processed1/"+str(i)+".png"
  width = 1024
  height = 1024
  type = 'png'
  ResizeImage(filein, fileout, width, height)

for j in range(404):
    Resize(j)