#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/18 16:13
# @Author  : zerone
# @Site    : 
# @File    : resize.py
# @Software: PyCharm
import os
import cv2 as cv
file_path = "H:/SFU-VOC_360/VOC_360/fisheye/"
file_path_1 = "H:/undistorted/"
for img in os.listdir(file_path):
    img_new = file_path_1 + img
    image = cv.imread(file_path + img)
    image_1 = cv.imread(file_path_1 + img)
    h, w, c = image.shape
    image = cv.resize(image_1, (h,w))
    cv.imwrite(img_new, image)
    print(img)