import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from retinanet import model as m
def load_classes(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1
        try:
            # class_name, class_id = row
            item = row[0]
            class_name = item.split(",")[0]
            class_id = item.split(",")[1]
        except ValueError:
            raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result

# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model_path, class_list):
    count = 0
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = m.resnet50(num_classes=20, pretrained=False)
    checkpoint = torch.load(model_path)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    '''
    model = torch.load(model_path)
    '''
    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    for img_name in os.listdir(image_path):
        count = count + 1
        image = cv2.imread(os.path.join(image_path, img_name))
        print(os.path.join(image_path, img_name))

        image_DDM = cv2.imread(os.path.join(image_path, img_name).replace("fisheye", "fisheye_1", 1))
        #image_undis = cv2.imread(os.path.join(image_path, img_name).replace("fisheye", "undistorted", 1))
        image_undis = cv2.imread(os.path.join("H:/undistorted", img_name))
        if image is None:
            continue
        image_orig = image.copy()
        image_orig_ddm = image_DDM.copy()
        image_orig_undis = image_undis.copy()
        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        image_ddm = cv2.resize(image_DDM, (int(round(cols * scale)), int(round((rows * scale)))))
        image_undis = cv2.resize(image_undis, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image_ddm = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image_undis = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)

        new_image[:rows, :cols, :] = image.astype(np.float32)
        new_image_ddm[:rows, :cols, :] = image_ddm.astype(np.float32)
        new_image_undis[:rows, :cols, :] = image_undis.astype(np.float32)

        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        image_ddm = new_image_ddm.astype(np.float32)
        image_ddm /= 255
        image_ddm -= [0.485, 0.456, 0.406]
        image_ddm /= [0.229, 0.224, 0.225]
        image_ddm = np.expand_dims(image_ddm, 0)
        image_ddm = np.transpose(image_ddm, (0, 3, 1, 2))

        image_undis = new_image_undis.astype(np.float32)
        image_undis /= 255
        image_undis -= [0.485, 0.456, 0.406]
        image_undis /= [0.229, 0.224, 0.225]
        image_undis = np.expand_dims(image_undis, 0)
        image_undis = np.transpose(image_undis, (0, 3, 1, 2))
        with torch.no_grad():

            image = torch.from_numpy(image)
            image_ddm = torch.from_numpy(image_ddm)
            image_undis = torch.from_numpy(image_undis)

            if torch.cuda.is_available():
                image = image.cuda()
                image_undis = image_undis.cuda()
                image_ddm = image_ddm.cuda()

            st = time.time()
            print(image.shape, image_orig.shape, scale)

            scores, classification, transformed_anchors = model((image.cuda().float(), image_ddm.cuda().float(), image_undis.cuda().float()))
            #scores, classification, transformed_anchors = model(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.3)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join("H:", "detection_2021_6_3", str(count) + ".jpg"), image_orig)
            print(count, "finished")
            #cv2.imshow('detections', image_orig)
            #cv2.waitKey(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')
    #csv_retinanet_change_35_0.19348881445990163_0.6652580300305362.pt
    parser.add_argument('--image_dir', default="H:/SFU-VOC_360/VOC_360/fisheye/", help='Path to directory containing images')
    parser.add_argument('--model_path', default="./csv_retinanet_undis_19_0.2291315644979477.pt", help='Path to model')
    parser.add_argument('--class_list', default="F:/VOC_label.csv", help='Path to CSV file listing class names (see README)')

    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.model_path, parser.class_list)
