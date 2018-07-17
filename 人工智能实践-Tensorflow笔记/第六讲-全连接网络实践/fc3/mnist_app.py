# coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(
                mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def pre_pic(picName):
    img = Image.open(picName)
    # 用消除锯齿的方法将 img resize 为28x28像素
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # 将图片变为灰度图，并转换为矩阵的形式赋值
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    # 反色，由于输入图片的影响
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            # 小于阈值的点认为是纯黑色 0
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            # 大于阈值的点认为是纯白色 255
            else:
                im_arr[i][j] = 255
    # 整理 im 形状为 1 行 784 列
    nm_arr = im_arr.reshape([1, 784])
    # 将 nm_arr 修改为浮点型
    nm_arr = nm_arr.astype(np.float32)
    # 再让现有的 RGB 图从 0~255 之间的数变为 0~1 间的浮点数
    img_ready = np.multiply(nm_arr, 1.0/255.0)
    # 这样完成了图片的预处理操作，符合神经网络对输入格式的要求
    return img_ready  # 整理好的待识别图片


def application():
    # 从控制台读入数字
    testNum = input("input the number of test pictures:")
    for i in range(testNum):
        # 从控制台读入字符串
        testPic = raw_input("the path of test picture:")
        testPicArr = pre_pic(testPic)
        # 喂入神经网络
        preValue = restore_model(testPicArr)
        print "The prediction number is:", preValue


def main():
    application()


if __name__ == '__main__':
    main()
