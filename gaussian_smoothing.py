import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaussian_kernel(size, sigma):
    kernel = cv.getGaussianKernel(size, sigma)
    kernel = np.outer(kernel, kernel)
    return kernel

def gaussian_filter(size, sigma):
    root = os.getcwd()
    imgPath = os.path.join(root, 'archive//Truck//Datacluster Truck (1).jpg')
    img = cv.imread(imgPath)

    fig = plt.figure()
    plt.subplot(121)
    kernel = gaussian_kernel(size, sigma)
    plt.imshow(kernel)

    ax = fig.add_subplot(1,2,2, projection="3d")
    x = np.arange(0,size,1)
    y = np.arange(0,size,1)
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,kernel,cmap='viridis')
        

    plt.show()

if __name__ == '__main__':
    gaussian_filter(32, 8)
