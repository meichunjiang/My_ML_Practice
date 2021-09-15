# coding UTF-8

# 基于Tensorflow和iris数据集实现前向传播、反向传播、可视化Loss曲线

import numpy as np
import tensorflow.compat.v1 as tf
from sklearn import datasets
from matplotlib import pyplot as plt

# Step 0 : 定义数据集

# Load datasets
x_data = datasets.load_iris().data              # 导入特征值数据（150个） - 4个特征值
y_data = datasets.load_iris().target            # 导入标签值数据（150个）
# 随机打乱数据。
np.random.seed(116)                             # 使用相同的随机种子，保证特征值和标签值在shuffle前后还能保持一一对应
np.random.shuffle(x_data)
np.random.seed(116)                             # 使用相同的随机种子，保证特征值和标签值在shuffle前后还能保持一一对应
np.random.shuffle(y_data)
np.random.seed(116)                             # 使用相同的随机种子，保证特征值和标签值在shuffle前后还能保持一一对应
# 拆分训练集和测试集
x_train = x_data[:-30]                          # 训练数据集 120个
y_train = y_data[:-30]                          # 训练数据集 120个
x_test  = x_data[-30:]                          # 测试数据集 30个
y_test  = x_data[-30:]                          # 测试数据集 30个
# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错 :tensorflow.python.framework.errors_impl.InvalidArgumentError: cannot compute MatMul as input #1(zero-based) was expected to be a double
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
# 训练数据打包
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)         # 每32对数据为一个包，待喂入神经网络
test_db  = tf.data.Dataset.from_tensor_slices((x_test ,y_test) ).batch(32)         # 每32对数据为一个包，待喂入神经网络



# Step 1 : Define the NN
# 输入层与数据集属性一致，为4个；输出层与数据集标签一致，为3个;
w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev = 0.1, seed = 1))         #
b1 = tf.Variable(tf.random.truncated_normal([3],stddev = 0.1, seed = 1))

# Step 3:  生成会话，训练模型
lr = 0.1
epoch = 500
loss_all = 0
test_acc = []                                                   # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
train_loss_results = []                                         # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据

for epoch in range(epoch):
    for step,(x_train,y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train,w1) + b1                      # 神经网络乘加运算
            y = tf.nn.softmax(y)                                # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            y_ = tf.one_hot(y_train,depth = 3)                  # 将标签值转换为独热码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_-y))              # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()                            # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确

        grads = tape.gradient(loss,[w1,b1])                     # 计算loss对各个参数的梯度

        w1.assign_sub(lr * grads[0])                            # w1自更新
        b1.assign_sub(lr * grads[1])                            # w1自更新

    print("Epoch {},loss : {}".format(epoch,loss_all / 4))      # 每个epoch，打印loss信息
    train_loss_results.append(loss_all / 4)                     # 将4个step的loss求平均记录在此变量中
    loss_all = 0                                                # loss_all归零，为记录下一个epoch的loss做准备

    # 测试部分
    total_correct, total_number = 0, 0                          # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1                          # 使用更新后的参数进行预测
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)                             # 返回y中最大值的索引，即预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype)                # 将pred转换为y_test的数据类型
        correct = tf.cast(tf.equal(pred,y_test),dtype=tf.int32) # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.reduce_sum(correct)                        # 将每个batch的correct数加起来
        total_correct += int(correct)                           # 将所有batch中的correct数加起来
        total_number += x_test.shape[0]                         # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
    acc = total_correct / total_number                          # 总的准确率等于total_correct/total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')                                # 图片标题
plt.xlabel('Epoch')                                             # x轴变量名称
plt.ylabel('Loss')                                              # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")                    # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()                                                    # 画出曲线图标
plt.show()                                                      # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')                                          # 图片标题
plt.xlabel('Epoch')                                             # x轴变量名称
plt.ylabel('Acc')                                               # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")                          # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()