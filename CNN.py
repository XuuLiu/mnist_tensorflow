import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('data/',one_hot=True)
trainimg=mnist.train.images
trainlabel=mnist.train.labels
testimg=mnist.test.images
testlabel=mnist.test.labels

n_input=784
n_output=10
stddev=0.1


weights={ # h,w,d,n
    'wc1':tf.Variable(tf.random_normal([3,3,1,64],stddev=stddev)),
    'wc2':tf.Variable(tf.random_normal([3,3,64,128],stddev=stddev)),
    'wd1': tf.Variable(tf.random_normal([7*7*128,1024], stddev=stddev)), #7*7*128图像大小，第一个全连接层
    'wd2':tf.Variable(tf.random_normal([1024,n_output],stddev=stddev))
}
biases={
    'bc1':tf.Variable(tf.zeros([64])),
    'bc2':tf.Variable(tf.zeros([128])),
    'bd1':tf.Variable(tf.zeros([1024])),
    'bd2': tf.Variable(tf.zeros([n_output]))
}

# 前向传播骨架
def conv_basic(_input,_w,_b,_keepratio):
    _input_r=tf.reshape(_input,shape=[-1,28,28,1]) #-1是说让他自动推断

    #conv layer1
    _conv1=tf.nn.conv2d(_input_r,_w['wc1'],strides=[1,1,1,1],padding='SAME')
    #strides针对不同维度定义,h和w改变，d和n上都为1就行，padding用0填充不够的
    _conv1=tf.nn.relu(tf.nn.bias_add(_conv1,_b['bc1']))
    _pool1=tf.nn.max_pool(_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    _pool_dr1=tf.nn.dropout(_pool1,_keepratio)

    #conv layer1
    _conv2=tf.nn.conv2d(_pool_dr1,_w['wc2'],strides=[1,1,1,1],padding='SAME')
    #strides针对不同维度定义,h和w改变，d和n上都为1就行，padding用0填充不够的
    _conv2=tf.nn.relu(tf.nn.bias_add(_conv2,_b['bc2']))
    _pool2=tf.nn.max_pool(_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    _pool_dr2=tf.nn.dropout(_pool2,_keepratio)

    _densel=tf.reshape(_pool_dr2,[-1,_w['wd1'].get_shape().as_list()[0]])

    #full connected layer1
    _fc1=tf.nn.relu(tf.add(tf.matmul(_densel,_w['wd1']),_b['bd1']))
    _fc_dr1=tf.nn.dropout(_fc1,_keepratio)

    #full connected layer2
    _out=tf.add(tf.matmul(_fc_dr1,_w['wd2']),_b['bd2'])

    out={
        'input_r':_input_r,
        'conv1':_conv1,
        'pool1':_pool1,
        'pool_dr1':_pool_dr1,
        'conv2':_conv2,
        'pool2':_pool2,
        'pool_dr2':_pool_dr2,
        'densel':_densel,
        'fc1':_fc1,
        'fc_dr1':_fc_dr1,
        'out':_out
    }
    return out

#占位
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_output])
keepratio=tf.placeholder(tf.float32)

#优化
_pred=conv_basic(x,weights,biases,keepratio)['out']
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred,y))
optm=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr=tf.equal(tf.arg_max(_pred,1),tf.arg_max(y,1))
accr=tf.reduce_mean(tf.cast(_corr,tf.float32))
init=tf.initialize_all_variables()

#迭代
sess=tf.Session()
sess.run(init)

training_epochs=50
batch_size=64
display_step=5

for epoch in range(training_epochs):
    avg_cost=0.
    #total_batch=int(mnist.train.num_examples/batch_size) #这个是对的，但是电脑不行就整个简单的跑跑看
    total_batch=10 #这个是不对的，都没有遍历完，缺失信息

    for i in range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(optm,feed_dict={x:batch_xs,y:batch_ys,keepratio:0.6})
        avg_cost+=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.})/total_batch

    if epoch % display_step==0:
        print "Epoch:%03d/%03d cost:%.9f" %(epoch,training_epochs,avg_cost)
        train_acc=sess.run(accr,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.})
        print "training accuracy: %.3f" %train_acc
        test_acc=sess.run(accr,feed_dict={x:testimg,y:testlabel,keepratio:1.})
        print "test accuracy: %.3f" %test_acc











