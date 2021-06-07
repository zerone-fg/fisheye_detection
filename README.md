# fisheye_detection
## retinanet文件夹
### model.py
改进之后的网络结构，涉及到三个分支的融合  
### dataloader.py  
数据集的加载
### utils.py  
对其中某些block的卷积层替换为可变形卷积  
## deform_conv2.py  
可变形卷积代码  
## train.py  
训练部分的代码  
## visualize_single_image.py  
单张图片检测效果可视化  
## voc_deal.py
对VOC360数据集做一定处理  
## visualize_feature.py  
某些层的特征图可视化
