# Author:LiPu
# coding=utf-8


import cv2
import os
import shutil
import argparse
from models import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cfg/visdrone.data', help='*.data file path')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--weights', type=str, default='./weights/yolov3', help='initial weights')

    parser.add_argument('--input_path', type=str, default='./data/input/', help='input_path')

    parser.add_argument('--iou_loss_thresh', type=int, default=0.5, help='iou_loss_thresh')
    parser.add_argument('--confidence_thresh', type=int, default=0.5, help='confidence_thresh')
    parser.add_argument('--nms_thresh', type=int, default=0.45, help='nms_thresh')

    opt = parser.parse_args()
    print(opt)
    num_class = len(utils.read_class_names(opt.data))
    classes = utils.read_class_names(opt.data)

    predicted_dir_path = './data/mAP/predicted'
    ground_truth_dir_path = './data/mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists("./data/detection/"): shutil.rmtree("./data/detection/")

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir("./data/detection/")

    # Build Model
    model = yolov3(opt)
    input_layer = tf.keras.layers.Input([opt.img_size, opt.img_size, 3])
    output = model(input_layer, training=False)
    model.load_weights("./weights/yolov3")
    labels = utils.read_label_path(opt.data, "test")
    images = utils.read_image_path(opt.data, "test")
    num = len(images)
    for i in range(num):
        image_path = images[i]
        label_path = labels[i]
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        f = open(label_path, 'r')
        txt = f.readlines()
        f.close()
        bbox_data_gt = np.array([list(map(float, line.strip("\n").split(' '))) for line in txt])

        if len(bbox_data_gt) == 0:
            bboxes_gt = []
            classes_gt = []
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, 1:5], bbox_data_gt[:, 0]
            bboxes_gt[:, 0], bboxes_gt[:, 2] = bboxes_gt[:, 0] * w, bboxes_gt[:, 2] * w
            bboxes_gt[:, 1], bboxes_gt[:, 3] = bboxes_gt[:, 1] * h, bboxes_gt[:, 3] * h
        ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

        print('=> ground truth of %s:' % image_name)
        num_bbox_gt = len(bboxes_gt)
        with open(ground_truth_path, 'w') as f:
            for i in range(num_bbox_gt):
                class_name = classes[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
        print('=> predict result of %s:' % image_name)
        predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
        # Predict Process
        image_size = image.shape[:2]
        image_data = utils.image_preporcess(np.copy(image), [opt.img_size, opt.img_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, opt.img_size, opt.confidence_thresh)
        bboxes = utils.nms(bboxes, opt.iou_loss_thresh, method='nms')

        image = utils.draw_bbox(image, bboxes, opt)
        cv2.imwrite("./data/detection/" + image_name, image)

        with open(predict_result_path, 'w') as f:
            for bbox in bboxes:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
