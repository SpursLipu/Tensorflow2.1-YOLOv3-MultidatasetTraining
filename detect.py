#! /usr/bin/env python
# coding=utf-8


import cv2
import argparse
from models import *
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cfg/visdrone.data', help='*.data file path')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--weights', type=str, default='./weights/init/yolov3.weights', help='initial weights')
    parser.add_argument('--iou_loss_thresh', type=int, default=0.5, help='iou_loss_thresh')

    opt = parser.parse_args()
    print(opt)

    visible_gpus = tf.config.experimental.list_physical_devices('GPU')
    print('Visible devices : ', visible_gpus)

    model = yolov3(opt)
    input_layer = tf.keras.layers.Input([opt.img_size, opt.img_size, 3])
    output = model(input_layer, training=False)
    # utils.load_weights(model, opt.weights)
    model.load_weights("./weights/yolov3")
    model.summary()

    image_path = "data/input/0000001_03499_d_0000006.jpg"

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(original_image), [opt.img_size, opt.img_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    pred_bbox = model(image_data, training=False)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, opt.img_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')

    image = utils.draw_bbox(original_image, bboxes, opt=opt, )
    image = Image.fromarray(image)
    image.save("data/output/output.jpg")
