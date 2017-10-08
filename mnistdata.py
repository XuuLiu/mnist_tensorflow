import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#下载数据集，存在当前新建的目录data下，采用01编码
mnist=input_data.read_data_sets('data/',one_hot=True)
print 'type of mnist is %s'%(type(mnist))
print 'number of train data is %d'%(mnist.train.num_examples)
print 'number of test data is %d'%(mnist.test.num_examples)


trainimg=mnist.train.images
trainlabel=mnist.train.labels
testimg=mnist.test.images
testlabel=mnist.test.labels

#取绘图样本
nsample=5
randidx=np.random.randint(trainimg.shape[0],size=nsample)
#将样本数据绘图
for i in randidx:
    curr_img=np.reshape(trainimg[i,:],(28,28))
    curr_label=np.argmax(trainlabel[i,:])
    plt.matshow(curr_img,cmap=plt.get_cmap('gray'))
    plt.title(""+str(i)+"th Training Data"
              +"Label is "+ str(curr_label))
    print ""+str(i)+"th Training Data"+"Label is"+str(curr_label)
    plt.show()

#batch的设置
batch_size=128
batch_xs, batch_ys=mnist.train.next_batch(batch_size)



