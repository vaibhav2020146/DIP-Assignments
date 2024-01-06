# First - order Techniques (Sobel)
from skimage.io import imread,imshow
import cv2
import numpy as np
img2=cv2.imread('C://Users//91991//Desktop//DIP//tank.jpg',0)
kernel = np.array([[0,0,0 ],
 [0,0,-1 ],
 [0,1 ,0]])
m,n=img2.shape
out=np.zeros([m,n])
for i in range(1,m-1):
 for j in range(1,n-1):
  temp=0
  for a in range(-1,2):
   for b in range(-1,2):
    temp+=img2[i+a,j+b]*kernel[a+1,b+1]
    out[i,j]=temp
m,n=img2.shape
out1=np.zeros([m,n])
for i in range(1,m-1):# this loop is used to traverse the image
 for j in range(1,n-1):# this loop is used to traverse the image
  temp=0
  for a in range(-1,2):#
   for b in range(-1,2):
    temp+=img2[i+a,j+b]*kernel[a+1,b+1]#multiplying the kernel with the image pixels so that we can get the desired output
    out1[i,j]=temp#storing the output in a variable
cv2.imshow('',out+out1)#adding the two outputs because in sobel technique we have to add the two outputs because we have to find the gradient of the image in both the directions 
#and for which we use kernel that is sa
cv2.waitKey()
cv2.destroyAllWindows()

#Robert Technique
kernel1 = np.array([[-1,-2,-1 ],
 [0,0,0],
 [1,2 ,1]])
m,n=img2.shape
out=np.zeros([m,n])
for i in range(1,m-1):
 for j in range(1,n-1):
  temp=0
  for a in range(-1,2):
   for b in range(-1,2):
    temp+=img2[i+a,j+b]*kernel[a+1,b+1]
    out[i,j]=temp
kernel2 = np.array([[-1,0,1 ],
 [-2,0,2 ],
 [1,0 ,1]])
m,n=img2.shape
out1=np.zeros([m,n])
for i in range(1,m-1):
 for j in range(1,n-1):
  temp=0
  for a in range(-1,2):
   for b in range(-1,2):
    temp+=img2[i+a,j+b]*kernel[a+1,b+1]
    out1[i,j]=temp
    dp=out+out1
cv2.imshow('',dp)
cv2.waitKey()
cv2.destroyAllWindows()

# Prewitt Technique
kernel1 = np.array([[1,1,1 ],
 [0,0,0],
 [-1,-1 ,-1]])
m,n=img2.shape
out=np.zeros([m,n])
for i in range(1,m-1):
 for j in range(1,n-1):
  temp=0
  for a in range(-1,2):
   for b in range(-1,2):
    temp+=img2[i+a,j+b]*kernel[a+1,b+1]
    out[i,j]=temp
kernel2 = np.array([[-1,0,1 ],
 [-1,0,1 ],
 [1,0 ,1]])
m,n=img2.shape
out1=np.zeros([m,n])
for i in range(1,m-1):
 for j in range(1,n-1):
  temp=0
  for a in range(-1,2):
   for b in range(-1,2):
    temp+=img2[i+a,j+b]*kernel[a+1,b+1]
    out1[i,j]=temp
    dp=out+out1
cv2.imshow('',dp)
cv2.waitKey()
cv2.destroyAllWindows()


#Second Order Techniques
# Unsharpening mask
kernel = np.array([[1/9,1/9,1/9],
 [1/9,1/9,1/9],
 [1/9,1/9,1/9]])
m,n=img2.shape
out=np.zeros([m,n])
for i in range(1,m-1):
 for j in range(1,n-1):
  temp=0
  for a in range(-1,2):
   for b in range(-1,2):
    temp+=img2[i+a,j+b]*kernel[a+1,1+b]#multiplying the kernel with the image pixels so that we can get the desired output
    out[i,j]=temp
    mean_im=img2 - out#subtracting the original image with the mean image
    final=mean_im + img2#adding the mean image with the original image
cv2.imshow('',final)
cv2.waitKey()
cv2.destroyAllWindows()

# Laplacian filter , Kernel size : 3x3
kernel = np.array([[0, -1, 0],
 [-1, 5,-1],
 [0, -1, 0]])
m,n=img2.shape
out=np.zeros([m,n])
for i in range(1,m-1):
 for j in range(1,n-1):
  temp=0
  for a in range(-1,2):
   for b in range(-1,2):
    temp+=img2[i+a,j+b]*kernel[a+1,1+b]
    out[i,j]=temp
cv2.imshow('',out)
cv2.waitKey()
cv2.destroyAllWindows()

# Laplacian filter, Non-zero diagonal elements 
kernel = np.array([[-1, -1, -1],#kernel for laplacian filter
 [-1, 9,-1],
 [-1, -1, -1]])
m,n=img2.shape
out=np.zeros([m,n])
for i in range(1,m-1):
 for j in range(1,n-1):
  temp=0
  for a in range(-1,2):
   for b in range(-1,2):
    temp+=img2[i+a,j+b]*kernel[a+1,1+b]
    out[i,j]=temp
cv2.imshow('',out)
cv2.waitKey()
cv2.destroyAllWindows()