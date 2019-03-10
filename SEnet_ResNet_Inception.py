# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:36:58 2018

@author: wgh
"""
#本框架的所有例如(a*b)*c备注 中的abc分别代表行，列和深度
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
def weights(shape):
    init=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)
#weights=tf.get_variable('name',[5,5,32,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
def bias(shape):
    init=tf.constant(0.1,dtype=tf.float32,shape=shape)
    return init

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def max_pool_same(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,1,1,1],padding="SAME")

def avg_pool_single(x):#[7,7]
    return tf.nn.avg_pool(x,ksize=[1,7,7,1],strides=[1,1,1,1],padding="VALID")
    
in_put=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
in_put_reshape=tf.reshape(in_put,[-1,28,28,1])
#the  conv1

conv1_weights=weights([3,3,1,8])
conv1_bias=bias([8])
conv1=conv2d(in_put_reshape,conv1_weights)
conv1_output=tf.nn.bias_add(conv1,conv1_bias)
#先不进行线性整流单元的加入  输出[28,28]*8

#the  conv2
conv2_weights=weights([3,3,8,16])
conv2_bias=bias([16])
conv2=conv2d(conv1_output,conv2_weights)
conv2_output=tf.nn.bias_add(conv2,conv2_bias)
#[28,28]*16

#the  conv3
conv3_weights=weights([3,3,16,32])
conv3_bias=bias([32])
conv3=conv2d(conv2_output,conv3_weights)
conv3_output=tf.nn.bias_add(conv3,conv3_bias)
#[28,28]*32

#the conv4
conv4_weights=weights([3,3,32,64])
conv4_bias=bias([64])
conv4=conv2d(conv3_output,conv4_weights)
relu4=tf.nn.relu(tf.nn.bias_add(conv4,conv4_bias))
conv4_output=max_pool(relu4)
#[14,14]*64

#the inceptionV1
inception1_conv1_weights=weights([1,1,64,16])#(1*1)*16
inception1_conv1_bias=bias([16])
inception1_conv1=conv2d(conv4_output,inception1_conv1_weights)
inception1_conv1_output=tf.nn.relu(tf.nn.bias_add(inception1_conv1,inception1_conv1_bias))
#[14*14]*16

inception1_conv2_weights=weights([3,3,64,16])#(3*3)*16
inception1_conv2_bias=bias([16])
inception1_conv2=conv2d(conv4_output,inception1_conv2_weights)
inception1_conv2_output=tf.nn.relu(tf.nn.bias_add(inception1_conv2,inception1_conv2_bias))
#[14*14]*16

inception1_conv3_weights=weights([5,5,64,32])#(5*5)*32
inception1_conv3_bias=bias([32])
inception1_conv3=conv2d(conv4_output,inception1_conv3_weights)
inception1_conv3_output=tf.nn.relu(tf.nn.bias_add(inception1_conv3,inception1_conv3_bias))
#[14*14]*32

#pool
inception1_pool_output=max_pool_same(tf.nn.relu(conv4_output))
#[14*14]*64

# concat inception_conv1
inception1_output=tf.concat([inception1_conv1_output,inception1_conv2_output,inception1_conv3_output,inception1_pool_output],3)
#[14*14]*128

#the conv5-14
conv5_weights=weights([3,3,128,128])
conv5_bias=bias([128])
conv5=conv2d(inception1_output,conv5_weights)
relu5=tf.nn.relu(tf.nn.bias_add(conv5,conv5_bias))

conv6_weights=weights([3,3,128,128])
conv6_bias=bias([128])
conv6=conv2d(relu5,conv6_weights)
relu6=tf.nn.relu(tf.nn.bias_add(conv6,conv6_bias))

conv7_weights=weights([3,3,128,128])
conv7_bias=bias([128])
conv7=conv2d(relu6,conv7_weights)
relu7=tf.nn.relu(tf.nn.bias_add(conv7,conv7_bias))
#ResNet1
Resnet5_7=relu5+relu7#此处的resnet网络结构 仅供参考  在更深的网络结构的后端使用resnet效果会比较好
#end
conv8_weights=weights([3,3,128,128])
conv8_bias=bias([128])
conv8=conv2d(Resnet5_7,conv8_weights)
relu8=tf.nn.relu(tf.nn.bias_add(conv8,conv8_bias))

conv9_weights=weights([3,3,128,128])
conv9_bias=bias([128])
conv9=conv2d(relu8,conv9_weights)
relu9=tf.nn.relu(tf.nn.bias_add(conv9,conv9_bias))

#ResNet2
Resnet5_7_9=relu5+relu7+relu9
#end

conv10_weights=weights([3,3,128,128])
conv10_bias=bias([128])
conv10=conv2d(Resnet5_7_9,conv10_weights)
relu10=tf.nn.relu(tf.nn.bias_add(conv10,conv10_bias))


conv11_weights=weights([3,3,128,128])
conv11_bias=bias([128])
conv11=conv2d(relu10,conv11_weights)
relu11=tf.nn.relu(tf.nn.bias_add(conv11,conv11_bias))

#ResNet3
Resnet5_7_9_11=relu5+relu7+relu9+relu11
#end

conv12_weights=weights([3,3,128,128])
conv12_bias=bias([128])
conv12=conv2d(Resnet5_7_9_11,conv12_weights)
relu12=tf.nn.relu(tf.nn.bias_add(conv12,conv12_bias))

conv13_weights=weights([3,3,128,128])
conv13_bias=bias([128])
conv13=conv2d(relu12,conv13_weights)
relu13=tf.nn.relu(tf.nn.bias_add(conv13,conv13_bias))

#ResNet4
Resnet5_7_9_11_13=relu5+relu7+relu9+relu11+relu13
#end

conv14_weights=weights([3,3,128,128])
conv14_bias=bias([128])
conv14=conv2d(Resnet5_7_9_11_13,conv14_weights)
relu14=tf.nn.relu(tf.nn.bias_add(conv14,conv14_bias))

#end the ten conv   output[14*14]*128

#pool layer
direct10_pool=max_pool(relu14)
#[7*7]*128

#THE SECOND DIRECT10 15-24
conv15_weights=weights([3,3,128,128])
conv15_bias=bias([128])
conv15=conv2d(direct10_pool,conv15_weights)
relu15=tf.nn.relu(tf.nn.bias_add(conv15,conv15_bias))

conv16_weights=weights([3,3,128,128])
conv16_bias=bias([128])
conv16=conv2d(relu15,conv16_weights)
relu16=tf.nn.relu(tf.nn.bias_add(conv16,conv16_bias))

conv17_weights=weights([3,3,128,128])
conv17_bias=bias([128])
conv17=conv2d(relu16,conv17_weights)
relu17=tf.nn.relu(tf.nn.bias_add(conv17,conv17_bias))

#ResNet5
Resnet15_17=relu15+relu17
#end

conv18_weights=weights([3,3,128,128])
conv18_bias=bias([128])
conv18=conv2d(Resnet15_17,conv18_weights)
relu18=tf.nn.relu(tf.nn.bias_add(conv18,conv18_bias))

conv19_weights=weights([3,3,128,128])
conv19_bias=bias([128])
conv19=conv2d(relu18,conv19_weights)
relu19=tf.nn.relu(tf.nn.bias_add(conv19,conv19_bias))

#ResNet6
Resnet15_17_19=relu15+relu17+relu19
#end

conv20_weights=weights([3,3,128,128])
conv20_bias=bias([128])
conv20=conv2d(Resnet15_17_19,conv20_weights)
relu20=tf.nn.relu(tf.nn.bias_add(conv20,conv20_bias))

conv21_weights=weights([3,3,128,128])
conv21_bias=bias([128])
conv21=conv2d(relu20,conv21_weights)
relu21=tf.nn.relu(tf.nn.bias_add(conv21,conv21_bias))

#ResNet7
Resnet15_17_19_21=relu15+relu17+relu19+relu21
#end

conv22_weights=weights([3,3,128,128])
conv22_bias=bias([128])
conv22=conv2d(Resnet15_17_19_21,conv22_weights)
relu22=tf.nn.relu(tf.nn.bias_add(conv22,conv22_bias))

conv23_weights=weights([3,3,128,128])
conv23_bias=bias([128])
conv23=conv2d(relu22,conv23_weights)
relu23=tf.nn.relu(tf.nn.bias_add(conv23,conv23_bias))

#ResNet7
Resnet15_17_19_21_23=relu15+relu17+relu19+relu21+relu23
#end

conv24_weights=weights([3,3,128,128])
conv24_bias=bias([128])
conv24=conv2d(Resnet15_17_19_21_23,conv24_weights)
relu24=tf.nn.relu(tf.nn.bias_add(conv24,conv24_bias))

#END THE SECOND DIRECT10
#output [7*7]*128

#SeNet
#SeNet实际上就是参数化 深度方向上的重要性 并通过卷积操作来进行权值更新

Squeeze=avg_pool_single(relu24)#[1*1]*128
num_FCconnect1=8
FCconnect_in=tf.reshape(Squeeze,[-1,1*1*128])
weights_FC=weights([1*1*128,num_FCconnect1])
bias_FC1=bias([num_FCconnect1])
FC1_out=tf.matmul(FCconnect_in,weights_FC)
relu_SE1=tf.nn.relu(tf.nn.bias_add(FC1_out,bias_FC1))#[1*1]*8

FC2_weights=weights([1*1*8,128])
bias_FC2=bias([128])
FC2_out=tf.matmul(relu_SE1,FC2_weights)
relu_SE2=tf.nn.relu(tf.nn.bias_add(FC2_out,bias_FC2))#[1*1]*128

Resize_SE2=tf.reshape(relu_SE2,[-1,1*1*128])
#sigmoid
output_list1 = []
output_list2 = []
#tensor无法直接赋值  引入中间的list
sigmoid=tf.nn.sigmoid(Resize_SE2)#50,[1*1]*128这里的50为batchsize

List=[]
ListAll=[]

for i in range(50):
    for j in range(128):
        List.append(relu24[i,:,:,j]*sigmoid[i,j])#在深度方向上加入权重
    outputs=tf.stack(List,2)
    ListAll.append(outputs)
    List=[]
outputs_SEnet=tf.stack(ListAll)
#end SeNet output50[7*7]128
        

#full_conneceted1
X_in=tf.reshape(outputs_SEnet,[-1,7*7*128])
fc1=weights([7*7*128,1024])
fc1_bias=bias([1024])
fc_1=tf.matmul(X_in,fc1)
relu3=tf.nn.relu(tf.nn.bias_add(fc_1,fc1_bias))

#dropout
keep_prob=tf.placeholder(tf.float32)
fc_1_leave=tf.nn.dropout(relu3,keep_prob)

#fullconnected2
fc2=weights([1024,10])
fc2_bias=bias([10])
y_conv=tf.nn.softmax(tf.nn.bias_add(tf.matmul(fc_1_leave,fc2),fc2_bias))


cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4,name='add').minimize(cross_entropy)
corret_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(corret_prediction,tf.float32))

#模型保存
Saver=tf.train.Saver()
initial=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(initial)
    for i in range(400):
        batch=mnist.train.next_batch(50)
        if i%100==0:
            train_accuracy=sess.run(accuracy,feed_dict={in_put:batch[0],y_:batch[1],keep_prob:1.0})
            print("step %d,training accurancy %g"%(i,train_accuracy))
        sess.run(train_step,feed_dict={in_put:batch[0],y_:batch[1],keep_prob:0.5})
    #Saver.save(sess,"Saver/Ha",global_step=1)
    file_writer = tf.summary.FileWriter('C://Users//wgh//Desktop//log', sess.graph)
   # graph_def = tf.get_default_graph().as_graph_def()
    #output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,graph_def, ['add'])
 
   # with tf.gfile.GFile("Saver/combined_model.pb", 'wb') as f:
   #     f.write(output_graph_def.SerializeToString())










