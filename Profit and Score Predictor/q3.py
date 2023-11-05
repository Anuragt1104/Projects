import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def normalization(X):
    m,n=X.shape
    for i in range(n):
        X=(X-np.mean(X,axis=0))/np.std(X,axis=0)
    return X

#Accessing data
train=pd.read_excel('q2train.xlsx')
test=pd.read_excel('q2test.xlsx')
aptitude=train['Aptitude']
verbal=train['Verbal']
Label=train['Label']
train=np.transpose(np.array([aptitude,verbal]))
Y=np.array(Label)

#Visualization of data
p_x=[]
p_y=[]
f_x=[]
f_y=[]
for i in range(len(Label)):
    if Label[i]==1:
        p_x+=[aptitude[i]]
        p_y+=[verbal[i]]
    else:
        f_x+=[aptitude[i]]
        f_y+=[verbal[i]]
plt.scatter(p_x,p_y,color='blue',label='pass')
plt.scatter(f_x,f_y,color='red',label='fail')
plt.xlabel('Aptitude')
plt.ylabel('Verbal')
plt.legend()
plt.show()

#Applying Logistic Regression on training data
print(train)
train=normalization(train)
m,n=train.shape
train=np.append(np.ones((m,1)),train,axis=-1)
Y=Y.reshape(m,1)
theta=np.random.rand(n+1,1) #Random Initialization
alpha=0.01
for i in range(2500):
    est=1/(1+np.exp(-np.matmul(train,theta)))
    grad=(1/m)*(np.dot(np.transpose(train),est-Y))
    theta=theta-(alpha*grad)
print(theta)

#Plotting decision boundary
p_x=[]
p_y=[]
f_x=[]
f_y=[]
for i in range(m):
    if Y[i,0]==1:
        p_x+=[train[i,1]]
        p_y+=[train[i,2]]
    else:
        f_x+=[train[i,1]]
        f_y+=[train[i,2]]
plt.scatter(p_x,p_y,color='blue',label='pass')
plt.scatter(f_x,f_y,color='red',label='fail')
plt.xlabel('Aptitude')
plt.ylabel('Verbal')
plt.legend()
#Boundary
xaxis = [-0.5,0.75]
slope = -theta[1]/theta[2]
c = (0.5-theta[0])/theta[2]
yaxis = slope*xaxis + c
plt.plot(xaxis,yaxis,color='green')
plt.title('Decision Boundary of Logistic Regression')
plt.show()

#Applying decision boundary obtained on test data
test=np.transpose(np.array([test['Aptitude'],test['Verbal']]))
test=normalization(test)
results=[]
for i in range(test.shape[0]):
    pred=1/(1+pow(math.e,(-(theta[0]+(theta[1]*test[i][0])+(theta[2]*test[i][1])))))
    if pred>=0.5:
        results+=[1]
    else:
        results+=[0]

#Writing in output file
with open('Output1.txt','w') as f:
    for items in results:
        f.write('%s\n'% items)