import cv2 as cv
import os
import random
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
root_dir = "H:/coco"
classes = {}
coco_labels = {}
coco_labels_inverse = {}
labels = {}
def coco_label_csv(set_name):
    coco = COCO(os.path.join(root_dir, 'annotations', 'instances_' + set_name + '.json'))
    categories = coco.loadCats(coco.getCatIds())
    categories.sort(key=lambda x: x['id'])
    csv_list = []
    for c in categories:
        coco_labels[len(classes)] = c['id']
        coco_labels_inverse[c['id']] = len(classes)
        classes[c['name']] = len(classes)
    for key, value in classes.items():
        labels[value] = key
        temp = key + "," + str(value)
        csv_list.append(temp)
    df = pd.DataFrame({"name": csv_list})
    df.to_csv("F:/coco_class_" + set_name + ".csv", index=False, header=False)
def coco_label_to_label(coco_label):
    return coco_labels_inverse[coco_label]
def load_annotations(coco, image_ids, image_index):
    annotations_ids = coco.getAnnIds(imgIds=image_ids[image_index], iscrowd=False)
    annotations = np.zeros((0, 5))
    if len(annotations_ids) == 0:
        return annotations
    coco_annotations = coco.loadAnns(annotations_ids)
    for idx, a in enumerate(coco_annotations):
        if a['bbox'][2] < 1 or a['bbox'][3] < 1:
            continue
        annotation = np.zeros((1, 5))
        annotation[0, :4] = a['bbox']
        annotation[0, 4] = coco_label_to_label(a['category_id'])
        annotations = np.append(annotations, annotation, axis=0)
    # transform from [x, y, w, h] to [x1, y1, x2, y2]
    annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
    annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
    return annotations
def coco_annotations_csv(set_name):
    coco = COCO(os.path.join(root_dir, 'annotations', 'instances_' + set_name + '.json'))
    image_ids = coco.getImgIds()
    df = []
    for image_index in range(len(image_ids)):
        image_info = coco.loadImgs(image_ids[image_index])[0]
        path = os.path.join(root_dir, 'images', set_name, image_info['file_name'])
        annotations = load_annotations(coco, image_ids, image_index)
        for index in range(annotations.shape[0]):
            temp = "{},{},{},{},{},{}".format(path, annotations[index][0], annotations[index][1], annotations[index][2], annotations[index][3], labels[annotations[index][4]])
            df.append(temp)
    dataframe = pd.DataFrame({'name': df})
    dataframe.to_csv("F:/coco_annotations_" + set_name + ".csv", index=False, header=False)
coco_label_csv('val2017')
coco_annotations_csv('val2017')