#coding:utf-8
#预测多或预测少的影响一样
#0导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
# 生成 32 行 2 列的数据集 X 包含 32组 0~1 之间的随机数 x1,x2
X = rdm.rand(32,2)
# 取出，每组 x1,x2 求和，再加上随机噪声构建标准答案 Y_
# .rand() 会生成 0~1 的前闭后开区间随机数，
# 除以10会变成 0到0.1之间的随机数
# -0.05变成 -0.05 到 +0.05 之间的随机数
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]

#1定义神经网络的输入、参数和输出，定义前向传播过程。
# 定义网络参数 w1
x = tf.placeholder(tf.float32, shape=(None, 2))
# 定义计算结果输出 y
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

#2定义损失函数及反向传播方法。
#定义损失函数为MSE,反向传播方法为梯度下降。
loss_mse = tf.reduce_mean(tf.square(y_ - y))
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse) 
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss_mse)
train_step = tf.train.AdamOptimizer(0.01).minimize(loss_mse)
# 让均方误差损失函数向减小的方向优化

#3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = (i*BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            print "After %d training steps, w1 is: " % (i)
            print sess.run(w1), "\n"
    print "Final w1 is: \n", sess.run(w1)
#在本代码#2中尝试其他反向传播方法，看对收敛速度的影响，把体会写到笔记中

