
# coding: utf-8

# In[1]:


import numpy as np #矩阵运算


# In[2]:



lists1=np.array([[0.42,-0.087,0.58],
                 [-0.2,-3.3,-3.4],
                 [1.3,-0.32,1.7],
                 [0.39,0.71,0.23],
                 [-1.6,-5.3,-0.15],
                 [-0.029,0.89,-4.7],
                 [-0.23,1.9,2.2],
                 [0.27,-0.3,-0.87],
                 [-1.9,0.76,-2.1],
                 [0.87,-1.0,-2.6]])

lists2=np.array([[-0.4,0.58,0.089],
                 [-0.31,0.27,-0.04],
                 [0.38,0.055,-0.035],
                 [-0.15,0.53,0.011],
                 [-0.35,0.47,0.034],
                 [0.17,0.69,0.1],
                 [-0.011,0.55,-0.18],
                 [-0.27,0.61,0.12],
                 [-0.065,0.49,0.0012],
                 [-0.12,0.054,-0.063]])

# lists3=np.array([[0.83,1.6,-0.014],
#                  [1.1,1.6,0.48],
#                  [-0.44,-0.41,0.32],
#                  [0.047,-0.45,1.4],
#                  [0.28,0.35,3.1],
#                  [-0.39,-0.48,0.11],
#                  [0.34,-0.079,0.14],
#                  [-0.3,-0.22,2.2],
#                  [1.1,1.2,-0.46],
#                  [0.18,-0.11,-0.49]])
# list_group = [lists1,lists2,lists3]


# #### 一维

# In[5]:


def getMean1(points):
    return np.mean(points)
def getVar1(points):
    u = getMean1(points)
    n = len(points)
    re = 0
    for i in points:
        re = re + (i-u)**2
    return re/n


# In[14]:


for i in range(3):
    print("特征x",i+1,"的均值μ的最大似然估计为 ",'%.8f' % getMean1(lists1[:,i]))
    print("特征x",i+1,"的方差σ^2的最大似然估计为   ",'%.8f' % getVar1(lists1[:,i]))


# In[8]:


#均值的最大似然估计为全体样本的平均值
def getMean(points):
#     输入为矩阵 求列均值 10x3
    return np.mean(points,0)
def getVar(points):
    u = getMean(points)
    n,d = points.shape
    re_matrix=np.zeros([d,d])
    for i in points:
        temp = np.matrix(i)-u
        re_matrix = re_matrix + np.dot(temp.T,temp)
    return re_matrix/n


# #### 二维

# In[13]:


print("特征x1,x2 的均值μ的最大似然估计为 \n", 
      getMean(lists1[:,[0,1]]))
print("特征x1,x2 的协方差矩阵σ^2的最大似然估计为 \n", getVar(lists1[:,[0,1]]))
print("特征x1,x3 的均值μ的最大似然估计为 \n", getMean(lists1[:,[0,2]]))
print("特征x1,x3 的协方差矩阵σ^2的最大似然估计为 \n", getVar(lists1[:,[0,2]]))
print("特征x2,x3 的均值μ的最大似然估计为 \n", getMean(lists1[:,[1,2]]))
print("特征x2,x3 的协方差矩阵σ^2的最大似然估计为 \n", getVar(lists1[:,[1,2]]))


# #### 三维

# In[15]:


print("w1的均值的最大似然估计为")
print(getMean(lists1))
print("w1协方差矩阵的最大似然估计为")
print(getVar(lists1))


# #### d

# In[18]:



print("w1的均值的最大似然估计为")
print(getMean(lists2))
for i in range(3):
    print("特征x",i+1,"的均值μ的最大似然估计为 ",'%.8f' % getVar1(lists2[:,i]))

