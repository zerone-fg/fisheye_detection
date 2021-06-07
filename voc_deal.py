try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os
import cv2 as cv
import pandas as pd
Annotation_path = "H:/SFU-VOC_360/VOC_360/Annotations/"
def visualize(path, boxes):
    img = cv.imread(path)
    for box in boxes:
        cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv.imshow("show", img)
    cv.waitKey(-1)
def GetAnnotBoxLoc(AnotPath):
    path = "H:/SFU-VOC_360/VOC_360/fisheye/" + AnotPath.split("/")[-1].replace(".xml", ".jpg")
    tree = ET.ElementTree(file=AnotPath)
    root = tree.getroot()
    ObjectSet = root.findall('object')
    ObjBndBoxSet = {}
    result = []
    vis = []
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        if BndBox!=None:
            x1 = int(BndBox.find('xmin').text)
            y1 = int(BndBox.find('ymin').text)
            x2 = int(BndBox.find('xmax').text)
            y2 = int(BndBox.find('ymax').text)
            result.append("{},{},{},{},{},{}".format(path, x1, y1, x2, y2, ObjName))
        #vis.append([x1, y1, x2, y2])
    #visualize(path, vis)
    return result
def csv_gen():
    set_name = ["train.txt", "val_1.txt"]
    file_dir = "H:/SFU-VOC_360/VOC_360/ImageSets/Main/"
    for set in set_name:
        df = []
        with open(file_dir + set, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                file = Annotation_path + line + ".xml"
                result = GetAnnotBoxLoc(file)
                df.extend(result)
                print(file + "finished")
        data_frame = pd.DataFrame({"name": df})
        data_frame.to_csv("F:/VOC_Annotations_{}.csv".format(set), index=False, header=False)
        f.close()
def voc_label():
    label = ["person", "diningtable", "chair", "tvmonitor", "sofa", "bicycle", "bottle", "aeroplane", "train", "car", "bird", "dog",
             "sheep", "horse", "cat", "motorbike", "bus", "pottedplant", "cow", "boat"]
    df = []
    for i, item in enumerate(label):
        line = "{},{}".format(item, i)
        df.append(line)
    data_frame = pd.DataFrame({"name": df})
    data_frame.to_csv("F:/VOC_label.csv", index=False, header=False)
csv_gen()


