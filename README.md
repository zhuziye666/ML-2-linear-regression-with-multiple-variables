# ML-2-linear-regression-with-multiple-variables
practise about machine learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#导入数据并进行预处理
path='ex1data2.txt'
data=pd.read_csv(path,header=None,names=['Size','Bedrooms','Price'])
mean=data.mean()
std=data.std()
print(mean[0])
data=(data-data.mean())/data.std()
print(data.head())
data.insert(0,'Ones',1)
cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
y=y*std[2]+mean[2]
print(X.head())
print(y.head())
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0,0]))
print(X.shape,y.shape,theta.shape)
#代价函数
def computeCost(X,y,theta):
    inner=np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
#梯度下降
def gradientDescent(X,y,theta,alpha,iters):
    m=len(y)
    J_history=np.zeros(iters)
    for i in range(iters):
        inner1=(X*theta.T-y).T*X
        theta=theta-(alpha/m*(inner1))
        J_history[i]=computeCost(X,y,theta)
    return theta,   J_history
#赋值，运行梯度下降求结果
alpha=0.03
iters=400
theta,j=gradientDescent(X,y,theta,alpha,iters)
print(theta,j[iters-1])
predict1=[1,(1650-mean[0])/std[0],(3-mean[1])/std[1]]*theta.T
print('For a 1650 sq-ft, 3 br house(using gradient descent),\
we predict a price: $',predict1)
#绘制梯度下降过程中，代价函数值变化过程
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),j,'r')
ax.set_xlabel('iterations')
ax.set_ylabel('J_cost')
ax.set_title("error vs. training epoch")
plt.show()
#使用正规方程法
def normalEqu(X,y):
    theta2=np.linalg.inv(X.T*X)*X.T*y
    return theta2
final_theta=normalEqu(X,y)
print(final_theta)
predict2=[1,(1650-mean[0])/std[0],(3-mean[1])/std[1]]*final_theta
print('For a 1650 sq-ft, 3 br house(using normal equation),\
we predict a price: $',predict2)
