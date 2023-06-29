# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         nms.py
# Description:  非极大值抑制算法
# Author:       Lv
# Date:         2023/6/28
# -------------------------------------------------------------------------------

"""
输入：给定所有可能的预测边框、给定的IoU阈值
输出：经过NMS算法过滤后的预测框
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class NMS:
    def __init__(self, center=False, scale=1.0):
        self.center = center
        self.scale = scale

    def compute_iou(self, bbox1, bbox2, eps=1e-8):
        """
        Compute the Iou of two bounding boxes
        :param bbox1:
        :param bbox2:
        :param eps:
        :return:
        """
        if self.center:
            x1, y1, w1, h1 = bbox1
            xmin1, ymin1 = int(x1 - w1 / 2.0), int(y1 - h1 / 2.0)
            xmax1, ymax1 = int(x1 + w1 / 2.0), int(y1 + h1 / 2.0)
            x2, y2, w2, h2 = bbox2
            xmin2, ymin2 = int(x2 - w2 / 2.0), int(y2 - h2 / 2.0)
            xmax2, ymax2 = int(x2 + w2 / 2.0), int(y2 + h2 / 2.0)
        else:
            xmin1, ymin1, xmax1, ymax1 = bbox1
            xmin2, ymin2, xmax2, ymax2 = bbox2

        # 计算交集对角坐标
        xx1 = np.max([xmin1, xmin2])
        yy1 = np.max([ymin1, ymin2])
        xx2 = np.min([xmax1, xmax2])
        yy2 = np.min([ymax1, ymax2])

        # 计算交集面积
        w = np.max([0.0, xx2 - xx1 + 1])
        h = np.max([0.0, yy2 - yy1 + 1])
        area_intersection = w * h

        # 计算并集面积
        area1 = (ymax1 - ymin1 + 1) * (xmax1 - xmin1 + 1)
        area2 = (ymax2 - ymin2 + 1) * (xmax2 - xmin2 + 1)
        area_union = area2 + area1 - area_intersection

        # 计算交并比
        iou = area_intersection / (area_union + eps)
        return iou

    @classmethod
    def nms(cls, dets, iou_threshold=0.5, score_threshold=0.5):
        """
        实施NMS
        :param dets:
        :param iou_threshold:
        :param score_threshold:
        :return:
        """
        dets = dets[np.where(dets[:, -1] >= score_threshold)[0]]  # 筛除score小于阈值
        print('dets:', dets)
        xmin = dets[:, 0]
        ymin = dets[:, 1]
        xmax = dets[:, 2]
        ymax = dets[:, 3]
        scores = dets[:, -1]

        order = scores.argsort()[::-1]  # 按score降序排列，argsort返回降序排序索引
        print('order:', order)
        areas = (xmax - xmin + 1) * (ymax - ymin + 1)  # 计算各GT面积
        result = []  # 保留最优结果

        while len(order) > 0:
            top_index = order[0]
            result.append(top_index)

            # 将最高分的GT和剩余GT比较
            xx1 = np.maximum(xmin[top_index], xmin[order[1:]])
            yy1 = np.maximum(ymin[top_index], ymin[order[1:]])
            xx2 = np.minimum(xmax[top_index], xmax[order[1:]])
            yy2 = np.minimum(ymax[top_index], ymax[order[1:]])

            # 计算交集
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            area_intersection = w * h
            # 计算并集
            print('areas[top_index]:', areas[top_index])
            area_union = areas[top_index] + areas[order[1:]] - area_intersection
            print('area_union:', area_union)
            # 计算Iou
            iou = area_intersection / area_union + 1e-8
            print('iou:', iou)
            inds = np.where(iou <= iou_threshold)[0]
            print('inds:', inds)
            order = order[inds + 1]
            print('order:', order)
        return result


if __name__ == "__main__":
    info = np.array([
        [30, 10, 200, 200, 0.95],
        [25, 15, 180, 220, 0.98],
        [35, 40, 190, 170, 0.96],
        [60, 60, 90, 90, 0.3],
        [20, 30, 40, 50, 0.1],
    ])
    NMS.nms(info)
    print(info.shape)
    print(info[:, -1])
