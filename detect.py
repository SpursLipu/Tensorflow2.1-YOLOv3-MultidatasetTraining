# coding=utf-8


import cv2
import argparse
from models import *
from PIL import Image
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cfg/mask.data', help='*.data file path')
    parser.add_argument('--img_size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--weights', type=str, default='./weights/darknet/yolov3', help='initial weights')

    parser.add_argument('--input_path', type=str, default='./data/input/', help='input_path')

    parser.add_argument('--iou_loss_thresh', type=int, default=0.5, help='iou_loss_thresh')
    parser.add_argument('--confidence_thresh', type=int, default=0.5, help='confidence_thresh')
    parser.add_argument('--nms_thresh', type=int, default=0.45, help='nms_thresh')
    parser.add_argument('--model', type=str, default='darknet', help='initial weights')

    opt = parser.parse_args()
    print(opt)

    visible_gpus = tf.config.experimental.list_physical_devices('GPU')
    print('Visible devices : ', visible_gpus)

    model = yolov3(opt)
    input_layer = tf.keras.layers.Input([opt.img_size, opt.img_size, 3])
    output = model(input_layer, training=False)
    if str(opt.weights).split('/')[2] == 'init':
        utils.load_weights(model, opt.weights)
    else:
        model.load_weights(opt.weights)
    model.summary()

    input_path = os.listdir(opt.input_path)
    for image_path in input_path:
        image_path = opt.input_path + image_path
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]

        image_data = utils.image_preporcess(np.copy(original_image), [opt.img_size, opt.img_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        pred_bbox = model(image_data, training=False)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, opt.img_size, opt.confidence_thresh)
        bboxes = utils.nms(bboxes, opt.nms_thresh, method='nms')

        image = utils.draw_bbox(original_image, bboxes, opt=opt, )
        image = Image.fromarray(image)
        image.save(image_path.replace('input', 'output'))
