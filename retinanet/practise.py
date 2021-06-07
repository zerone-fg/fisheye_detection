#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/4 19:57
# @Author  : zerone
# @Site    : 
# @File    : practise.py
# @Software: PyCharm
import cv2
import os
for img in os.listdir("E:/fisheye_1/"):
    img_new = cv2.resize(cv2.imread("E:/fisheye_1/" + img), (128, 128), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("E:/fisheye_1/" + img, img_new)