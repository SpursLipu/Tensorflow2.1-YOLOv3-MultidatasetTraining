# Author:LiPu
import argparse
from models import *
from dataset import Dataset
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_loss(opt, pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        optimizer.lr.assign(opt.lr_init)
        tf.print("Epoch:%d   STEP:%d/%d   lr: %.5f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (epoch, global_steps, steps_per_epoch, optimizer.lr.numpy(),
                                                           giou_loss, conf_loss,
                                                           prob_loss, total_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data', type=str, default='cfg/visdrone.data', help='*.data file path')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--weights', type=str, default='./weights/init/yolov3.weights', help='initial weights')
    parser.add_argument('--lr_init', type=int, default=1e-4, help='initial lr')
    parser.add_argument('--iou_loss_thresh', type=int, default=0.5, help='iou_loss_thresh')
    parser.add_argument('--conti', type=bool, default=False, help='continue training')

    opt = parser.parse_args()
    print(opt)

    visible_gpus = tf.config.experimental.list_physical_devices('GPU')
    print('Visible devices : ', visible_gpus)

    trainset = Dataset(opt, 'train')
    steps_per_epoch = len(trainset)
    total_steps = opt.epochs * steps_per_epoch
    global_steps = 0
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = yolov3(opt)
    input_layer = tf.keras.layers.Input([opt.img_size, opt.img_size, 3])
    output = model(input_layer, training=True)
    if opt.conti == True:
        model.load_weights("./weights/yolov3")
    else:
        utils.load_weights(model, "./weights/init/yolov3.weights")
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(opt.epochs):
        for image_data, target in trainset:
            train_step(image_data, target)
            global_steps += 1
    model.save_weights("./weights/yolov3")
    global_steps = 0
