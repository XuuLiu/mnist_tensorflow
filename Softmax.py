import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#读取数据
mnist=input_data.read_data_sets('data/',one_hot=True)
trainimg=mnist.train.images
trainlabel=mnist.train.labels
testimg=mnist.test.images
testlabel=mnist.test.labels

print trainimg.shape
print trainlabel.shape
print testimg.shape
print testlabel.shape

x=tf.placeholder("float",[None,784]) #先占地[无穷大，trainimg的第二维]
y=tf.placeholder("float",[None,10])
W=tf.Variable(tf.zeros([784,10]))#784个像素点，需要784个权重参数，10为我们的输出
b=tf.Variable(tf.zeros([10]))

#逻辑回归
actv=tf.nn.softmax(tf.matmul(x,W)+b) #逻辑回归为二分类的任务，利用softmax将其升级为多分类的任务
#softmax 的输入为对一个样本来说，对每一个分类的得分值

#损失函数为-logP，P为属于真实的label的概率值，
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv),reduction_indices=1))

#loss梯度下降优化
learning_rate=0.01
optm=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#测试
pred=tf.equal(tf.arg_max(actv,1), tf.arg_max(y,1)) #预测值类别和真实值类别是否相等，true 或 false
#准确度
accr=tf.reduce_mean(tf.cast(pred,'float'))#把true 转为1或 false转为0

init=tf.initialize_all_variables()#变量初始化


training_epochs=50 #所有样本迭代次数
batch_size=128 #每进行一次迭代要用多少个样本
display_step=5

sess=tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost=0.
    num_batch=int(mnist.train.num_examples/batch_size)
    for i in range(num_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(optm,feed_dict={x:batch_xs,y:batch_ys}) #梯度下降求解，place_holder导进来
        feeds={x:batch_xs,y:batch_ys}
        avg_cost+=sess.run(cost,feed_dict=feeds)/num_batch

    if epoch %display_step==0:
        feeds_train={x:batch_xs,y:batch_ys}
        feeds_test={x:mnist.test.images,y:mnist.test.labels}
        train_acc=sess.run(accr,feed_dict=feeds_train)
        test_acc=sess.run(accr,feed_dict=feeds_test)
        print "Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f" %(epoch,training_epochs,avg_cost,train_acc,test_acc)

