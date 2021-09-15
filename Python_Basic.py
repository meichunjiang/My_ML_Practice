# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import tensorflow as tf

import tensorflow.compat.v1.keras.layers as keraslayers
from tensorflow.compat.v1.train import Saver

# 载入MNIST数据集
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# 搭建模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练并验证模型
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)





class Basic_datatype:
    def __init__(self):
        print("I am going to study Python 3.0 from https://www.runoob.com/python3/python3-intro.html!")
    def Basic_Grammar(self):
        print("Python Basic Grammar.")
    def PrintTest(self):
        str = 'This is a String !'
        str_width = int(2*len(str))

        print(str.ljust(str_width,'-'))         # ljust(str,width)
        print(str.center(str_width,'@'))        # center(str,width)
        print(str.rjust(str_width,'#'))         # rjust(str,width)

        print(format(str,'<30'))                # left  :<
        print(format(str, '>30'))               # right :>
        print(format(str, '^30'))               # center:^
    def Py_OperatorTest(self):
        print("\n01 Arithmetic Operator Demo")
        print('     -Basic Arithmetic Operator 1+2*3/4 = ',1+2*3/4)
        print('     -Basic Arithmetic Operator 5%2 = ', 5%2)                    # 取余
        print('     -Basic Arithmetic Operator 5//3 = ', 5//3)                  # 向下取整

        print("\n02 逻辑比较 Operator Demo")
        print(' == ， ！= ， <, >, <= ,>= return True or False ')

        print("\n03 赋值 Operator Demo")
        print('     c += a  -->  c = c + a ')
        print('     c -= a  -->  c = c - a ')
        print('     c *= a  -->  c = c * a ')
        print('     c /= a  -->  c = c / a ')
        print('     c %= a  -->  c = c % a ')
        print('     c := a  -->  Walrus Operator ，可在表达式内部为变量赋值 after Python3.8')

        # 场景1
        my_list = [1, 2, 3]
        count = len(my_list)
        if count >= 3:                      print(f"        Error, {count} is too many items")
        if (count := len(my_list)) >= 3:    print(f"        Error, {count} is too many items")          # when converting to walrus operator...

        # 场景2
        f = open(os.getcwd() +'/data.xml')
        line = f.readline()
        while line:
            line = f.readline()

        while line := f.readline():  print(line)         # when converting to walrus operator...


        print('\n04 Python逻辑运算符')
        print('     and / or / not')

        print('\n05 Python成员运算符')
        print('    in / not in')

        print('\n06 Python身份运算符')                   # is 用于判断两个变量引用 {对象} 是否为同一个， == 用于判断引用变量的 {值} 是否相等。
        print('    is / is not ')

        a,b = 20,20
        if (a is b):            print("1 - a 和 b 有  相同的标识")
        else:                   print("1 - a 和 b 没有相同的标识")
        if (id(a) == id(b)):    print("2 - a 和 b 有  相同的标识")
        else:                   print("2 - a 和 b 没有相同的标识")

        b = 30                      # 修改变量 b 的值
        if (a is b):            print("3 - a 和 b 有相同的标识")
        else:                   print("3 - a 和 b 没有相同的标识")
        if (a is not b):        print("4 - a 和 b 没有相同的标识")
        else:                   print("4 - a 和 b 有相同的标识")

        a = [1, 2, 3]
        b = a
        if (a is b):            print(" a and b are same object")
        else:                   print(" a and b are NOT same object")
        if (a == b):            print(" a and b have same value")
        else:                   print(" a and b have NOT same value")

        b = a[:]                # system allocate memory to b
        if (a is b):            print(" a and b are same object")
        else:                   print(" a and b are NOT same object")
        if (a == b):            print(" a and b have same value")
        else:                   print(" a and b have NOT same value")
    def Py_Number(self):
        print('可以使用 ** 操作来进行幂运算: 5的平方(5**2) = ', 5**2)
        print('在交互模式中，最后被输出的表达式结果被赋值给变量 _')
        tax = 12.5 / 100
        price = 100.50
        print('price * tax = ', price * tax )
        # print('_ is ',_)
        b = price+_
        print('price + _ = ',price+_)
        print(round(_, 2))
        return
    def Py_String(self):
        return
    def Py_List(self):
        return
    def Py_Tuple(self):
        return
    def Py_Dictionary(self):
        return
    def Py_Set(self):
        return
bd = Basic_datatype()
bd.Basic_Grammar()
bd.PrintTest()
bd.Py_OperatorTest()
# bd.Py_Number()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/