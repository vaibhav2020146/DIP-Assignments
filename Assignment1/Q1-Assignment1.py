import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage.io import imread,imshow


# Read image from location C:\Users\91991\Desktop\DIP
img = cv2.imread('C://Users//91991//Desktop//DIP//image1.jpg',0)

# Display image
#cv2.imshow('', img)

#hold image for 5 seconds
#cv2.waitKey(5000) 


#implement 3x3 gaussian filter:
#kernel size
k = 1/3

for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        temp = img[i-1,j-1]*k + img[i-1,j]*k + img[i-1,j+1]*k + img[i,j-1]*k + img[i,j]*k + img[i,j+1]*k + img[i+1,j-1]*k + img[i+1,j]*k + img[i+1,j+1]*k
        img[i,j] = temp

# Display image
cv2.imshow('', img)

#hold image for 5 seconds
cv2.waitKey(5000)



# Median filter with kernel size of 3x3
'''m, n = img.shape
img_new1 = np.zeros([m, n])
for i in range(1, m-1):
  for j in range(1, n-1):
    temp=[]
    for a in range(-1,2):
      for b in range(-1,2):
        temp.append(img[i+a,j+b])
        temp = sorted(temp)
        img_new1[i, j]= temp[4] 
cv2.imshow(img_new1)


# Median 5x5
output3 = np.zeros([m, n])
for i in range(2, m-2):
  for j in range(2, n-2):
    temp=[]
    for a in range(-2,3):
      for b in range(-2,3):
        temp.append(img[i+a,j+b])
        temp = sorted(temp)
        output3[i, j]= temp[12]

cv2.imshow(output3'''