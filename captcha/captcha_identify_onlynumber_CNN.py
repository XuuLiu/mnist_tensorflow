import tensorflow as tf
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


number=['0','1','2','3','4','5','6','7','8','9']
#alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#ALPHABET=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def random_captcha_text(char_set=number,captcha_size=4):
    captcha_text=[]
    for i in range(captcha_size):
        c=random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

def gen_captcha_text_and_image():
    image=ImageCaptcha()

    captcha_text=random_captcha_text()
    captcha_text=''.join(captcha_text) # list to string

    captcha=image.generate(captcha_text)

    captcha_image=Image.open(captcha)
    captcha_image=np.array(captcha_image)
    return captcha_text, captcha_image

def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img

def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('Max length is %d' %(MAX_CAPTCHA))

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + int(c) # find the place where is 1
        vector[idx] = 1
    return vector


def vec2text(vec):
    text = []
    char_pos = vec.nonzero()[0] #location of where is 1
    for i, c in enumerate(char_pos):
        number = i % 10
        text.append(str(number))

    return "".join(text)

# create the  frame of cnn
def crack_captcha_cnn(w_alpha=0.01,b_alpha=0.1): # initialize w and b with a small number
    x=tf.reshape(X,shape=[-1,IMAGE_HEIGHT,IMAGE_WIDTH,1]) #[batch_size,length,width,channel]

    w_c1=tf.Variable(w_alpha*tf.random_normal(([3,3,1,32]))) #filter size [length, width, depth, number]
    b_c1=tf.Variable(b_alpha*tf.random_normal([32]))
    conv1=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1)) #relu(wx+b)
    conv1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #pooling mainly max_pooling
    # ksize=[1,2,2,1] is using a 2*2 size window
    conv1=tf.nn.dropout(conv1,keep_prob=keep_prob)

    w_c2=tf.Variable(w_alpha*tf.random_normal(([3,3,32,64]))) #filter size [length, width, depth, number]
    b_c2=tf.Variable(b_alpha*tf.random_normal([64]))
    conv2=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2))
    conv2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2=tf.nn.dropout(conv2,keep_prob=keep_prob)

    w_c3=tf.Variable(w_alpha*tf.random_normal(([3,3,64,64]))) #filter size [length, width, depth, number]
    b_c3=tf.Variable(b_alpha*tf.random_normal([64]))
    conv3=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2,w_c3,strides=[1,1,1,1],padding='SAME'),b_c3))
    conv3=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2=tf.nn.dropout(conv3,keep_prob=keep_prob)

    # full connected layer
    w_d=tf.Variable(w_alpha*tf.random_normal([8*20*64,1024])) #8*20 is calculated thought all those conv and poolings
    b_d=tf.Variable(b_alpha*tf.random_normal([1024]))
    dense=tf.reshape(conv3,[-1,w_d.get_shape().as_list()[0]]) # to value of [8*20*64,1024]
    dense=tf.nn.relu(tf.add(tf.matmul(dense,w_d),b_d))
    dense=tf.nn.dropout(dense,keep_prob)

    w_out= tf.Variable(w_alpha*tf.random_normal([1024,MAX_CAPTCHA*CHAR_SET_LEN]))
    b_out=tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
    out=tf.add(tf.matmul(dense,w_out),b_out)
    return out


def get_next_batch(batch_size=128):
    batch_x=np.zeros([batch_size,IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y=np.zeros([batch_size,MAX_CAPTCHA*CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while True:
            text,image=gen_captcha_text_and_image()
            if image.shape==(60,160,3):
                return text,image

    for i in range(batch_size):
        text,image=wrap_gen_captcha_text_and_image()
        image =convert2gray(image)

        batch_x[i,:]=image.flatten() # make it to 1 dimension
        batch_y[i,:]=text2vec(text) # to one_hot encoding

    return batch_x,batch_y



def train_crack_captcha_cnn():
    output=crack_captcha_cnn()
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    predict=tf.reshape(output,[-1,MAX_CAPTCHA,CHAR_SET_LEN])
    max_idx_p=tf.argmax(predict,2)
    max_idx_1=tf.argmax(tf.reshape(Y,[-1,MAX_CAPTCHA,CHAR_SET_LEN]),2)
    correct_pred=tf.equal(max_idx_p,max_idx_1)
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    saver=tf.train.Saver() # save model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step=0
        while True:
            batch_x,batch_y=get_next_batch(64)
            _,loss_=sess.run([optimizer,loss],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75}) # keep_prob for  dropout layer
            print step, loss_

            if step %100==0:
                batch_x_test,batch_y_test=get_next_batch(100)
                acc=sess.run(accuracy,feed_dict={X:batch_x_test,Y:batch_y_test,keep_prob:0.75})
                print step,acc

                if acc>0.85: #when to save model, text acc>0.85
                    saver.save(sess,"./model/crack_captcha.model",global_step=step) #sava all session
                    break

        step+=1

def crack_captcha(captcha_image):
    output=crack_captcha_cnn()

    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,"./model/crack_capcha.model-810") # read the model that you saved while training

        predict=tf.argmax(tf.reshape(output,[-1,MAX_CAPTCHA,CHAR_SET_LEN]),2)
        text_list=sess.run(predict,feed_dict={X:[captcha_image],keep_prob:1})
        text=text_list[0].tolist()
        return text


if __name__=='__main__':
    train=0 # 0 is for training and 1 is for test, change it by yourself.
    if train==0:
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        text,image=gen_captcha_text_and_image()

        IMAGE_HEIGHT=60
        IMAGE_WIDTH=160
        MAX_CAPTCHA=len(text)

        chat_set=number # choice number of one position
        CHAR_SET_LEN=len(chat_set) # total position

        X=tf.placeholder(tf.float32,[None,IMAGE_HEIGHT*IMAGE_WIDTH]) #use gray pics
        Y=tf.placeholder(tf.float32,[None,MAX_CAPTCHA*CHAR_SET_LEN]) #lenth of label
        keep_prob=tf.placeholder(tf.float32) #dropout layer, keep ratio

        train_crack_captcha_cnn()
    if train==1:
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        IMAGE_HEIGHT=60
        IMAGE_WIDTH=160
        chat_set=number
        CHAR_SET_LEN=len(chat_set)

        text,image=gen_captcha_text_and_image()

        f=plt.figure()
        ax=f.add_subplot(111)
        ax.text(0.1,0.9,text,ha='center',va='center',transform=ax.transAxes)
        plt.imshow(image)

        plt.show()
        MAX_CAPTCHA=len(text)
        image=convert2gray(image)
        image=image.flatten()/255

        X=tf.placeholder(tf.float32,[None,IMAGE_WIDTH*IMAGE_HEIGHT])
        Y=tf.placeholder(tf.float32,[None,MAX_CAPTCHA*CHAR_SET_LEN])
        keep_prob=tf.placeholder(tf.float32)

        predict_text=crack_captcha(image)
        print "correct:("+text+") predict:("+str(predict_text)[1:len(str(predict_text)) - 1]+")"












