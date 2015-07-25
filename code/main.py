# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:43:24 2015

@author: Troy Kling
"""

import os
import cv2
import Image
import numpy as np
import matplotlib.pyplot as plt

os.chdir("C:/Users/Troy/Documents/Kaggle/DenoisingDirtyDocuments/Data/train")

img = np.array(Image.open("2.png"))
img_denoised = cv2.fastNlMeansDenoising(img, h = 25)
img_diff = img - img_denoised

figsize = (img.shape[0]/25, img.shape[1]/25)
plt.figure(figsize = figsize)
plt.imshow(img, cmap = plt.get_cmap("gray"))
plt.figure(figsize = figsize)
plt.imshow(img_denoised, cmap = plt.get_cmap("gray"))
plt.figure(figsize = figsize)
plt.imshow(img_diff, cmap = plt.get_cmap("gray"))
