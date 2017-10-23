import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets("data/",one_hot=True)
trainimgs,trainlabels,testimgs,testlabels=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
ntrain,ntest,dim,nclasses=trainimgs.shape[0],testimgs.shape[0],trainimgs.shape[1],trainlabels.shape[1]


diminput=28 #图像是28*28的,所以每一次输入为图像的一行，为1*28
dimhidden=128
dimoutput=nclasses
nsteps=28 #输入28次1*28的行
weights={
    'hidden':tf.Variable(tf.random_normal([diminput,dimhidden])),
    'out':tf.Variable(tf.random_normal([dimhidden,dimoutput]))
}
biases={
    'hidden':tf.Variable(tf.random_normal([dimhidden])),
    'out':tf.Variable(tf.random_normal([dimoutput]))
}

def _RNN(_X,_W,_b,_nsteps,_name):
    # 1. permute input from [batch_size,nstep,diminput] to [nsteps,batch_size,diminput]
    _X=tf.transpose(_X,[1,0,2])
    # 2. reshape input to [nsteps*batch_size,diminput]
    _X=tf.reshape(_X,[-1,diminput])
    # 3.Input layer => hidden layer
    _H=tf.matmul(_X,_W['hidden'])+_b['hidden']
    # 4.splite data to 'nsteps' chunks, an i-th chunck indicateds i-th batch data
    _Hsplit=tf.split(_H,_nsteps,0)
    # 5. get LSTM's final output(_LSTM_O) and state(_LSTM_S)
    # Both _LSTM_O and _LSTM_S consist of 'batch_size' elements
    # only _LSTM_O with be used to predict the output
    with tf.variable_scope(_name) as scope:
        #scope.reuse_variables() #变量共享
        lstm_cell=tf.contrib.rnn.BasicLSTMCell(dimhidden,forget_bias=1.0)
        _LSTM_O,_LSTM_S=tf.contrib.rnn.static_rnn(cell=lstm_cell,inputs=_Hsplit,dtype=tf.float32)
    # 6. output
    _O=tf.matmul(_LSTM_O[-1],_W['out'])+_b['out'] #-1是最后一个
    # return
    return {
        'X':_X,
        'H':_H,
        'Hsplit':_Hsplit,
        'LSTM_O':_LSTM_O,
        'LSTM_S':_LSTM_S,
        'O':_O
    }

learning_rate=0.001
x=tf.placeholder('float',[None,nsteps,diminput])
y=tf.placeholder('float',[None,dimoutput])
myrnn=_RNN(x,weights,biases,nsteps,'basic')
pred=myrnn['O']
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optm=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
accr=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),tf.float32))
init=tf.global_variables_initializer()

training_epoche=50
batch_size=16
display_step=5
sess=tf.Session()
sess.run(init)

for epoch in range(training_epoche):
    avg_cost=0.
    total_batch=int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape((batch_size,nsteps,diminput))
        feeds={x:batch_xs,y:batch_ys}
        sess.run(optm,feed_dict=feeds)
        avg_cost+=sess.run(cost,feed_dict=feeds)/total_batch

    if epoch%display_step==0:
        print "Epoch:%03d/%03d cost: %.9f"%(epoch,training_epoche,avg_cost)
        feeds={x:batch_xs,y:batch_ys}
        train_acc=sess.run(accr,feed_dict=feeds)
        print "Training accuracy:%.3f"%(train_acc)
        testimgs=testimgs.reshape((ntest,nsteps,diminput))
        feeds={x:testimgs,y:testlabels}
        test_acc=sess.run(accr,feed_dict=feeds)
        print "Test accuracy:%.3f"%(test_acc)





