
# coding: utf-8

# In[6]:


import math
import numpy as np
import random
random.seed(0)


# In[7]:


# 加载数据
import os
import struct

def load_mnist_train(path):
    labels_path = os.path.join(path,'train-labels.idx1-ubyte')
    images_path = os.path.join(path, 'train-images.idx3-ubyte')
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels
def load_mnist_test(path):
    labels_path = os.path.join(path,'t10k-labels.idx1-ubyte')
    images_path = os.path.join(path, 't10k-images.idx3-ubyte')
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels
X_train,Y_train = load_mnist_train('E:\学习\大三\机器学习\MNIST')
X_test,Y_test = load_mnist_test('E:\学习\大三\机器学习\MNIST')


# In[8]:


def tanh(x):
    return (np.exp(x)-np.exp(-1*x))/(np.exp(x)+np.exp(-1*x))
def dtanh(y):
    return 1.0 - y*y

class NN:
    def __init__(self, ni, nh_li, no):
        # 输入层、隐藏层列表、输出层
        self.ni = ni 
        self.nh_li = nh_li
        self.nh_len = len(nh_li)
        self.no = no
        self.w = []#输入层到第一层隐含层，。。。，最后一层隐含层到输出层 总长度为 nh_len+1
        
        self.w.append(np.random.normal(0.0, pow(self.nh_li[0],-0.5), (self.nh_li[0],self.ni)))
        for i in range(self.nh_len-1):
            self.w.append(np.random.normal(0.0, pow(self.nh_li[i+1],-0.5), (self.nh_li[i+1],self.nh_li[i])))
        self.w.append(np.random.normal(0.0, pow(self.no,-0.5), (self.no,self.nh_li[self.nh_len-1])))
        
    def oneRound(self, inputs,targets,N):
        targets = np.array(targets,ndmin=2).T
        # 激活输入层
        inputs = np.array(inputs,ndmin=2).T/255
        # 隐含层输出

        outputs = []# 输入层到第一个隐含层，。。。，最后一个隐含层到输出层  总长度为 nh_len+1
        outputs.append(tanh(np.dot(self.w[0],inputs)))
        for i in range(self.nh_len-1):
            outputs.append(tanh(np.dot(self.w[i+1],outputs[i])))
        # 输出层输出
        outputs.append(tanh(np.dot(self.w[self.nh_len], outputs[self.nh_len-1])))
        
        
        deltas = []#输出层，最后一个隐含层 。。。 第一个隐含层  总长度为 nh_len+1
        #计算输出层的误差
        output_deltas = dtanh(outputs[self.nh_len])*(targets - outputs[self.nh_len])
        deltas.append(output_deltas)
        for i in range(self.nh_len):
            deltas.append(dtanh(outputs[self.nh_len-1-i])* np.dot(self.w[self.nh_len-i].T, deltas[i]))

            
        for i in range(self.nh_len):
            self.w[self.nh_len-i] += N*np.dot(deltas[i], np.transpose(outputs[self.nh_len-i-1]))
        self.w[0] += N*np.dot(deltas[self.nh_len], np.transpose(inputs))
   
        return outputs[self.nh_len]
  
    def test(self, inputs_list,Y):
        le = len(Y)
        ans = 0
        for i in range(le):
            inputs = np.array(inputs_list[i],ndmin=2).T/255
            outputs = []# 输入层到第一个隐含层，。。。，最后一个隐含层到输出层  总长度为 nh_len+1
            outputs.append(tanh(np.dot(self.w[0],inputs)))
            for j in range(self.nh_len-1):
                outputs.append(tanh(np.dot(self.w[j+1],outputs[j])))
            # 输出层输出
            outputs.append(tanh(np.dot(self.w[self.nh_len], outputs[self.nh_len-1])))
            if np.argmax(outputs[self.nh_len])==Y[i]:
                ans+=1
        print('准确率',ans/le)
        return ans/le    

    def weights(self):
        return self.w

    def train(self, X, Y,iterations=100, N=0.01):
        len_total = len(Y)
        for i in range(iterations):
            print("第",i,"轮")
            for p in range(len_total):              
                inputs = X[p]
                target = Y[p]
                targets = trans(target)
                self.oneRound(inputs,targets,N)

    
    
def trans(li):
    re_list =[0.0]*10
    re_list[li]=1
    return re_list
                


# In[9]:


num=60000
n = NN(784, [100,40], 10)
accuracy_rate_2 = n.train(X_train[:num],Y_train[:num],iterations=10)
n.test(X_test,Y_test)


# In[10]:


num=60000
n = NN(784, [100,50,20], 10)
accuracy_rate = n.train(X_train[:num],Y_train[:num],iterations=200)

