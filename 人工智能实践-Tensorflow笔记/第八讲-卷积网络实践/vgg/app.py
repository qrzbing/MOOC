# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import vgg16
import utils
from Nclasses import labels

# 待识别图片的路径
img_path = raw_input('Input the path and image name:')
# 对待识别图片进行预处理
img_ready = utils.load_image(img_path)
# 打印img_ready的维度
# print("img_ready shape", tf.Session().run(tf.shape(img_ready)))
# 理论上输出 [1 224 224 3]

# 打印柱状图
fig=plt.figure(u"Top-5 预测结果") 

with tf.Session() as sess:
    # 给输入的特征图片占位
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    # 实例化 vgg
    vgg = vgg16.Vgg16() 
    # 运行前向神经网络结构，复现神经网络结构
    vgg.forward(images) 
    # 将待识别图像作为输入喂入计算 softmax的节点vgg.prob
    # 网络的输出probability就是通过vgg16网络的前向传播过程
    # 预测出的一千个分类的概率分布
    probability = sess.run(vgg.prob, feed_dict={images:img_ready})
    # 把probability列表中概率最高的五个所对应的probabilty列表的
    # 索引值存入Top 5。probabilty中的这5个索引值就是Nclasses.py
    # 标签字典中的键
    top5 = np.argsort(probability[0])[-1:-6:-1]
    print "top5:",top5
    # 新建 values 列表用来存储probability中元素的值
    # probability 元素的值是 top5 中5个物种出现的概率
    values = []
    # 新建bar_label列表用来存储标签字典中对应的值也就是5个物种的名称
    bar_label = []
    for n, i in enumerate(top5): 
        print "n:",n
        print "i:",i
        # probability中的元素也就是每个物种出现的概率
        values.append(probability[0][i]) 
        bar_label.append(labels[i]) 
        print i, ":", labels[i], "----", utils.percent(probability[0][i]) 
        
    # 构建一行一列的子图，画出第一张子图
    ax = fig.add_subplot(111) 
    ax.bar(
        range(len(values)), # 下标
        values, # 高度
        tick_label = bar_label, # 柱子的标签
        width = 0.5, # 柱子的宽度
        fc = 'g' # 柱子的颜色
    )
    ax.set_ylabel(u'probabilityit') 
    ax.set_title(u'Top-5') 
    # 在每个柱子的顶端添加对应的预测概率
    # a, b表示x, y坐标
    for a,b in zip(range(len(values)), values):
        ax.text(
            a,
            b + 0.0005, # 表示放在柱子顶部 0.005 高度的位置
            utils.percent(b), # 以百分比的形式表示
            ha = 'center', # 文本位于柱子顶端水平方向的居中位置
            va = 'bottom', # 文本水平放置在柱子顶端垂直方向上底端位置
            fontsize=7
        )   
    plt.show() 



