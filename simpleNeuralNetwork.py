import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('data/',one_hot=True)

#神经网络的结构定义
n_hidden_1=256 #第一层隐藏层有多少个神经元
n_hidden_2=128 #第二层隐藏层有多少个神经元
n_input=784 # 像素点的个数
n_classes=10 #最终分类的类别

#占位，样本个数不确定，用None代替
x=tf.placeholder("float",[None, n_input])
y=tf.placeholder("float",[None, n_classes])

#参数初始化
stddev=0.1
weights={
    'w1':tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),
    'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))
}
biases={
    'b1':tf.Variable(tf.zeros([n_hidden_1])),
    'b2':tf.Variable(tf.zeros([n_hidden_2])),
    'out':tf.Variable(tf.zeros([n_classes]))
}

#前向传播
def multilayer_perceptron(_X, _weights, _biases):
    layer_1=tf.nn.relu(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    layer_2=tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2']))
    return (tf.matmul(layer_2,_weights['out'])+_biases['out'])

#预测
pred=multilayer_perceptron(x,weights,biases)

#计算损失且梯度下降优化
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y)) #交叉熵损失函数,所有batch的平均一下
optm=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
# 准确率
corr=tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
accr=tf.reduce_mean(tf.cast(corr,'float'))

#超参数
training_epochs=50
batch_size=128
display_step=4

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

#运行
for epoch in range(training_epochs):
    avg_cost=0.
    total_batch=int(mnist.train.num_examples/batch_size)
    #迭代
    for i in range(total_batch):
        batch_xs, batch_ys=mnist.train.next_batch(batch_size)
        feeds={x:batch_xs,y:batch_ys}
        sess.run(optm,feed_dict=feeds)
        avg_cost+=sess.run(cost,feed_dict=feeds)
    avg_cost=avg_cost/total_batch
    #打印
    if (epoch+1)%display_step==0:
        print "Epoch:%03d/%03d cost: %.9f"%(epoch, training_epochs,avg_cost)
        feeds={x:batch_xs,y:batch_ys}
        train_acc=sess.run(accr,feed_dict=feeds)
        print "train accuracy: %.3f" %(train_acc)
        feeds={x:mnist.test.images, y:mnist.test.labels}
        test_acc=sess.run(accr,feed_dict=feeds)
        print "test accuracy: %.3f" %(test_acc)


