#coding:utf8

"""
http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
"""

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


import tensorflow as tf

# 定义计算模型
x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

# 成本函数是 定义“交叉熵”（cross-entropy）模型
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#　反向传播算法(backpropagation algorithm)
# 用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# 然后开始训练模型，这里我们让模型循环训练1000次！
# 使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练。
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 跑测试集
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


