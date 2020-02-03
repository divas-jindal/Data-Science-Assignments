import numpy as np
import pylab as pb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt


def normalize(x,y):
    #NORMALIZE USING Z SCORE
    for i in range(1,x.shape[1]):
        arr=x[:,i:i+1]
        mu=np.mean(arr)
        std=np.std(arr)
        for j in range(0,arr.shape[0]):
            arr[j]=(arr[j]-mu)/std
        x[:,i:i+1]=arr
    # y_mu=np.mean(y)
    # y_std=np.std(y) 
    # for i in range(0,y.shape[0]):
    #     y[i]=(y[i]-y_mu)/y_std
    return x,y   

def cost_function(x,y,theta):
    c=np.sum(np.square(x@(theta)-y))
    c=c/(2.0*y.size)
    return c

def rcost_function(x,y,theta,lamda):
    m=y.shape[0]
    e=np.dot(x,theta)
    e=e**2
    e=e/(2*m)
    s=np.sum(e)
    th=theta**2
    th=lamda*(th/(2*m))
    s2=np.sum(th)
    return s+s2

def predict_value(x,theta):
    return x@theta
    
def gradient_descent(x,y,alpha,num_iter):
    theta=np.ones((x[0].size,1))
    cost=[]
    for i in range(0,num_iter,1):
        h=predict_value(x,theta)
        theta=theta - ((alpha/y.size)*np.sum(np.transpose(h-y).dot(x)))
        cost.append(cost_function(x,y,theta))
    return theta,cost

def rgradient_descent(x,y,alpha,itr,lamda):
    theta=np.ones(x.shape[1])
    theta=np.reshape(theta,(-1,1))
    print(theta.shape)
    m=y.shape[0]
    f1=1
    r1=(alpha*lamda)/m
    f1=f1-r1
    cost=[]
    for i in range(0,itr):
        xt=np.transpose(x)
        h=x.dot(theta)
        h=h-y
        f2=xt.dot(h)
        theta=theta*(f1)-(alpha/m)*f2
        cost.append(rcost_function(x,y,theta,lamda))
    return theta,cost

def rms_error(x_test,y_test,theta):
    check=np.zeros((x_test.shape[0],1))
    check=predict_value(x_test,theta)
    leng=float(x_test.shape[0])
    rms_error=np.sum(np.square(check-y_test))
    rms_error/=leng
    rms_error=rms_error**(1/2)
    print('rms_error: ',rms_error)


def average_error(x_test,y_test,theta):
    check=np.zeros((x_test.shape[0],1))
    check=predict_value(x_test,theta)
    leng=float(x_test.shape[0])
    avg_error=(np.sum(check-y_test))/leng
    print('avg_error: ',avg_error)
    
def cont(x,y,th):
    th1=np.linspace(-1000,1000,num=1000)
    th2=np.linspace(-1000,1000,num=1000)
    TH1,TH2=np.meshgrid(th1,th2)
    th_copy=th.copy()
    jv=np.zeros((len(th1),len(th2)))
    for i in range(0,len(th1)):
        th_copy[1][0]=th1[i]
        for j in range(0,len(th2)):
            th_copy[2][0]=th2[j]
            jv[i][j]=cost_function(x,y,th_copy)
    plt.contour(TH1,TH2,jv)
    plt.savefig('contour_plots(with_regularization).png')

def rcont(x,y,th,lambd):
    th1=np.linspace(-1000,1000,num=1000)
    th2=np.linspace(-1000,1000,num=1000)
    TH1,TH2=np.meshgrid(th1,th2)
    th_copy=th.copy()
    jv=np.zeros((len(th1),len(th2)))
    for i in range(0,len(th1)):
        th_copy[1][0]=th1[i]
        for j in range(0,len(th2)):
            th_copy[2][0]=th2[j]
            jv[i][j]=rcost_function(x,y,th_copy,lamd)
    plt.contour(TH1,TH2,jv)
    plt.savefig('contour_plots(with_regularization).png')
    
#load datas
#get x and y matrix
#define value of m
'''
filename=input("enter the filename : ")

data=pd.read_csv(filename)

print(data.shape)

data=data.replace(np.nan,0)

x=data.iloc[:,:-1].values
y=data.iloc[:,data.shape[1]-1:].values

m=data.shape[0]
n=data.shape[1]

print(x.shape)
print(y.shape)
'''

boston=load_boston()
x=boston.data
y=boston.target
x=np.append(np.ones((x.shape[0],1)),x,1)
y=np.reshape(y,(-1,1))
m,n=x.shape
print(x.shape)
print(y.shape)

x,y=normalize(x,y)

train_start=0
train_end=int(.8*m)

test_start=int(.8*m)
test_end=m

x_train=x[train_start:train_end,:]
y_train=y[train_start:train_end]

x_test=x[test_start:test_end,:]
y_test=y[test_start:test_end]

alpha=0.0001
numIter=2000
lamd=10000

# print(x_train.shape,y_train.shape)
# print(x_test.shape,y_test.shape)
# print(x_train,y_train)
# print(x_train[0])

iterator=[i for i in range(0,numIter)]
# Uncomment to run with applying regulaization
theta,cost=rgradient_descent(x_train,y_train,alpha,numIter,lamd)

#theta,cost=gradient_descent(x_train,y_train,alpha,numIter)

# print(x.shape,y.shape)
# print(theta.shape)

rms_error(x_test,y_test,theta)
average_error(x_test,y_test,theta)

plt.plot(iterator,cost)
plt.title("cost function vs iteration")
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
plt.savefig('cost_function_vs_no_of_itr(with_regularization).png')

y_pred=predict_value(x_test,theta)
print("prediction shape")
print(y_pred.shape)

rcont(x_test,y_test,theta,lamd)
print("hi!")


#gradient discent with regularization
# cost_function_vs_no_of_itr(without_regularization)
