# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         frame_difference.py
# Description:  计算两帧图像之间的差值
# Author:       Lv
# Date:         2023/6/27
# -------------------------------------------------------------------------------

import cv2
import numpy as np

frame1_dir = './pictures/indoor_bright/000000.jpg'
frame2_dir = './pictures/indoor_bright/000001.jpg'

# 读取视频帧
frame1 = cv2.imread(frame1_dir)
frame2 = cv2.imread(frame2_dir)

# 将图像帧转换为灰度图像
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# 计算两个灰度图像的差值
diff = cv2.absdiff(gray1, gray2)

# 设置高斯核的大小与标准差
kernel_size = (5, 5)
sigma = 0

diff_gaussian = cv2.GaussianBlur(diff, kernel_size, sigma)

# 创建一个新窗口
cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
# 显示差值图像
cv2.imshow('Images', np.hstack((diff, diff_gaussian)))
cv2.waitKey(0)
cv2.destroyAllWindows()
