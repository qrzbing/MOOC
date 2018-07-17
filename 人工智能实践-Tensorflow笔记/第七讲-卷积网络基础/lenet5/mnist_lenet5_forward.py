#coding:utf-8
import tensorflow as tf
IMAGE_SIZE = 28
NUM_CHANNELS = 1
# 第一层卷积核的大小
CONV1_SIZE = 5
# 第一层卷积核的个数
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
# 第一层全连接网络神经元的个数
FC_SIZE = 512
# 第二层全连接网络神经元的个数
OUTPUT_NODE = 10

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None:
        tf.add_to_collection(
            'losses',
            tf.contrib.layers.l2_regularizer(regularizer)(w)
        ) 
    return w

def get_bias(shape): 
    b = tf.Variable(tf.zeros(shape))  
    return b

# 卷积计算函数
# w: 所用卷积核
def conv2d(x, w):  
    return tf.nn.conv2d(
        x,
        w,
        strides=[1, 1, 1, 1],
        padding='SAME' # 0填充
    )

# 最大池化计算函数
def max_pool_2x2(x):  
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    ) 

def forward(x, train, regularizer):
    # 第一层卷积
    # 初始化第一层卷积核
    conv1_w = get_weight(
        [CONV1_SIZE,CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM],
        regularizer
    )
    # 初始化第一层偏置
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    # 执行卷积计算，输入为图片x, 初始化的卷积核 conv1_w
    conv1 = conv2d(x, conv1_w) 
    # 激活函数
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # 最大池化
    pool1 = max_pool_2x2(relu1) 

    conv2_w = get_weight(
        [CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],
        regularizer
    ) 
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w) 
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    # 第二层卷积的输出，需要把它从三维张量变为二维张量
    pool2 = max_pool_2x2(relu2)

    # 得到pool2输出矩阵的维度存入 list 中
    pool_shape = pool2.get_shape().as_list()
    # 分别提取特征的长度、宽度、深度并相乘得到所有特征点的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 将 pool2 表示成 batch 行(pool_shape[0])，
    # 所有特征点作为个数列的二维形状，再喂入到全连接网络中
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE]) 
    # 将上层的输出乘本层线上的权重加上偏置过激活函数relu
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # 如果是训练阶段，则对该层的输出使用 50% 的dropout
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    # 通过第二层的全连接网络初始化第二层全连接网络的w、b
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    # 上层的输出和本层线上的w相乘加上偏置得到输出y
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y 
