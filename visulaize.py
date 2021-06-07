#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/3 20:27
# @Author  : zerone
# @Site    : 
# @File    : visulaize.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import math
def visualize_feature_map_sum(item,name):
    '''    将每张子图进行相加
    :param feature_batch:
    :return:
     '''
    feature_map = item.squeeze(0)
    c = item.shape[1]
    print(feature_map.shape)
    feature_map_combination=[]
    for i in range(0, c):
        feature_map_split = feature_map.data.cpu().numpy()[i, :, :]
        feature_map_combination.append(feature_map_split)
        feature_map_sum = sum(one for one in feature_map_combination)
        # feature_map = np.squeeze(feature_batch,axis=0)
    plt.figure()
    plt.title("combine figure")
    plt.imshow(feature_map_sum)
    plt.savefig('E:/feature_map_sum_a_'+name+'.png') # 保存图像到本地
    plt.show()
def visual(all_dict):
    exact_list = ['out_1']
    outputs = []
    for item in exact_list:
        x = all_dict[item]
        outputs.append(x)      # 特征输出可视化
    x = outputs
    k = 0
    print(x[0].shape[1])
    for item in x:
        c = item.shape[1]
        plt.figure()
        name = exact_list[k]
        plt.suptitle(name)
        '''
        for i in range(20):
            wid = math.ceil(math.sqrt(c))
            ax = plt.subplot(wid, wid, i + 1)
            ax.set_title('Feature {}'.format(i))
            ax.axis('off')
            figure_map = item.data.cpu().numpy()[0, i, :, :]
            plt.imshow(figure_map, cmap='jet')
            plt.savefig('E:/feature_map_aa_' + name + '.png')  # 保存图像到本地
        '''
        visualize_feature_map_sum(item, name)
        k = k + 1
    plt.show()