#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path='./mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path='./mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train='./data/mnist_train.tfrecords'
image_test_path='./mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path='./mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test='./data/mnist_test.tfrecords'
data_path='./data'
resize_height = 28
resize_width = 28

def write_tfRecord(tfRecordName, image_path, label_path):
    # 该函数接收到 
    # 参数1: 存放 tfRecords 文件的路径和文件名
    # 参数2: 图像路径
    # 参数3: 标签文件路径

    # 创建一个 writer
    writer = tf.python_io.TFRecordWriter(tfRecordName)  
    # 为了显示进度创建的计数器
    num_pic = 0
    # 以读的形式打开标签文件(为一个 txt 文件，每行有两组数据)
    f = open(label_path, 'r')
    # 读取整个文件的内容
    contents = f.readlines()
    f.close()
    # 遍历每行的内容
    for content in contents:
        # 用空格分割每行的内容，分割后组成列表 value
        value = content.split()
        # 图片文件的路径
        img_path = image_path + value[0]
        # 打开图片 
        img = Image.open(img_path)
        # 转换 img 为二进制数据
        img_raw = img.tobytes()
        # label 的每个元素赋值为 0 
        labels = [0] * 10
        # labels 对应的标签位赋值为 1
        labels[int(value[1])] = 1  
        # 创建 example
        # 封装
        example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                }))
        # 将 example 序列化
        writer.write(example.SerializeToString())
        num_pic += 1 
        if num_pic % 100 == 0:
            print ("[+] number of picture finished:", num_pic)
    writer.close()
    print("[+] write tfrecord successful")

def generate_tfRecord():
    # 判断保存路径是否存在
    isExists = os.path.exists(data_path) 
    if not isExists: 
        # 创建文件夹
        os.makedirs(data_path)
        print 'The directory was created successfully'
    else:
        print 'directory already exists'
    # 用下述自定义函数将训练集中的图片和标签生成名叫 tfRecord_train 的 tfRecord 文件 
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    # 生成测试集中的相应 tfRecord 文件
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)

def read_tfRecord(tfRecord_path):
    # 新建文件名队列
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
    reader = tf.TFRecordReader()
    # 将样本保存到 serialized_example 中进行解序列化
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           # 标签这里需要给出几分类。手写数字是 10 分类
                                        'label': tf.FixedLenFeature([10], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string)
                                        })
    # 将 'img_raw' 字符串转换成8位无符号整型
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 将形状变为 1行784列 
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label 

# 批读取 tfRecords 代码
def get_tfrecord(num, isTrain=True):
    # 实现了批获取训练集或测试集中的图片和标签

    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    # tf.train.shuffle_batch 函数
    # 从总样本中顺序取出 capacity 组数据，打乱顺序
    # 每次输出 batch_size 组
    # 如果 capacity 少于 min_after_dequeue
    # 从总样本中取数据填满 capacity
    # 整个过程使用了两个线程
    # 此时返回的图片和标签就是随机抽取出的 batch_size 组数据了
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size = num,
                                                    num_threads = 2,
                                                    capacity = 1000,
                                                    min_after_dequeue = 700)
    return img_batch, label_batch

def main():
    # 初始化数据集
    generate_tfRecord()

if __name__ == '__main__':
    main()
