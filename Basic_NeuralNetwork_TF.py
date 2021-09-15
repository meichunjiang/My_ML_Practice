# coding:UTF-8

# 基于Tensorflow实现一个基本神经网络训练过程、前向传播、反向传播、损失函数、学习率等基础概念(两个输入，一个输出，一个包含三个节点的隐藏层)
# 数据描述：自定义一个数据集，eg.生产一批零件将体积x1和重量x2为特征输入NeuralNetwork，通过NN后出入一个数值
# Input  Layer : X = [x1,x2]
# Hidden Layer : A = [a11,a12,a13]      # A = X*W_1, 为1X3的矩阵
# Output Layer : Y = [y]
#
# W_1 = [ [w11_1,w12_1,w13_1],
#         [w21_1,w22_1,w23_1]]          # 2X3矩阵
# W_2 = [[w11_2],
#        [w21_2],
#        [w31_2]]                       # 3X1矩阵
#
# A = X * W1
#   a11 = x1*w11_1 + x2*w21_1
#   a12 = x1*w12_1 + x2*w22_1
#   a13 = x1*w13_1 + x2*w23_1

# Y = A * W_2
#   y = a11*w11_2 + a12*w21_2 + a13*w31_2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


seed = 23455
BATCH_SIZE = 8
DATASET_NUMBER = 32

# Step 0 : 定义数据集
X = np.random.RandomState(seed).rand(DATASET_NUMBER,2)      # 基于seed产生随机数，并返回32行2列的矩阵
Y = [[int(x0+x1<1)] for(x0,x1) in X]            # 自定义标签集合，即，体积和质量的和小于1，零件为合格品。（人为自定义的标准，无实际意义）
print(X)
print(Y)

# Step 1 ： 定义神经网络的输入、参数和输出，定义前向传播过程。
x  = tf.placeholder(tf.float32, shape = (None,2))        # None 为组数，由于当前未知，用None占位 ；2 为特征数(即 质量和体积)
y_ = tf.placeholder(tf.float32, shape = (None,1))        # None 为组数，由于当前未知，用None占位 ；1 为x的对应结果

w1 = tf.Variable(tf.random_normal([2,3],stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1, seed = 1))

a = tf.matmul(x,w1)                                     # 前向传播过程的公式描述, 用矩阵乘法来实现
y = tf.matmul(a,w2)                                     # 前向传播过程的公式描述，用矩阵乘法来实现

# Step 2 ： 定义损失函数及反向传播方法
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)        # 梯度下降优化法，学习率为0.001
# train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# Step 3:  生成会话，训练STEPS
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(w1)
    print(w2)

    # 训练模型
    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % DATASET_NUMBER
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})


        if i%50 ==0:
            total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
            print("After %d training steps,loss on all data is %g"%(i,total_loss))

    print('\n')
    print('w1:\n',sess.run(w1))
    print('w2:\n',sess.run(w2))