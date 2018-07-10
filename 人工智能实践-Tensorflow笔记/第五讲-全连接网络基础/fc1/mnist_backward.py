# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

BATCH_SIZE = 200 # 定义每轮喂入的图片数
LEARNING_RATE_BASE = 0.1 # 最开始的学习率
LEARNING_RATE_DECAY = 0.99 # 学习率衰减率
REGULARIZER = 0.0001 # 正则化系数
STEPS = 50000 # 训练轮数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率
MODEL_SAVE_PATH="./model/" # 模型的保存路径
MODEL_NAME="mnist_model" # 模型的保存文件名

# 读入 mnist 
def backward(mnist):
    # 给
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    # 调用前向传播的程序计算输出 y
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    # 调用包含正则化的损失函数 loss
    loss = cem + tf.add_n(tf.get_collection('losses'))
    # 定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)
    # 定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 定义滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    # 实例化 saver
    saver = tf.train.Saver()
    # 
    with tf.Session() as sess:
        # 初始化所有变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 迭代 STEPS 轮
        for i in range(STEPS):
            # 从训练集中随机抽取 BATCH_SIZE 组数据和标签，把它们喂入神经网络执行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            # 每一千轮后输出 loss 值
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)

if __name__ == '__main__':
    main()


