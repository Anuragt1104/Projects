import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso

#Accessing data from sheet
data=pd.read_excel('PL1.xlsx')
X_train=np.array(data["Population in 10,000's"][:89])
Y_train=np.array(data["Profit In Lakhs(Rs)"][:89])
X_test=np.array(data["Population in 10,000's"][90:])
Y_test=np.array(data["Profit In Lakhs(Rs)"][90:])

#Data visualization
plt.scatter(X_train,Y_train,color='blue')
plt.xlabel("Population in 10,000's")
plt.ylabel("Profit in Lakhs(Rs)")
plt.show()

#Batch gradient descent
m=len(X_train)
converged=False
iterations=0
theta0=np.random.random(1)
theta1=np.random.random(1)
alpha=0.001
J = (0.5/m)*sum([(theta0 + theta1*X_train[i] - Y_train[i])**2 for i in range(m)]) #Cost function
while not converged:
    grad0=0
    grad1=0
    for i in range(m):
        grad0-=Y_train[i]-theta0-(theta1*X_train[i])
        grad1-=(Y_train[i]-(theta0+(theta1*X_train[i])))*X_train[i]
    theta0-=alpha*grad0/m
    theta1-=alpha*grad1/m
    err=(0.5/m)*sum([(theta0 + theta1*X_train[i] - Y_train[i])**2 for i in range(m)])
    #print(err)
    if abs(J-err)<0.0001:
        print('Converged')
        converged=True
    J=err
    iterations+=1
    if iterations==10000:
        print('Max. iterations reached')
        converged=True
print('For Batch Gradient Descent')
print(theta0,theta1,J,iterations)
plt.scatter(X_train,Y_train,color='blue')
plt.plot([min(X_train),max(X_train)],[theta0+theta1*min(X_train),theta0+theta1*max(X_train)],color='red')
plt.xlabel("Population in 10,000's")
plt.ylabel("Profit in Lakhs(Rs)")
plt.title('Batch Gradient Descent')
plt.show()

#Stochastic gradient descent
converged=False
iterations=0
theta0=np.random.random(1)
theta1=np.random.random(1)
J = (0.5/m)*sum([(theta0 + theta1*X_train[i] - Y_train[i])**2 for i in range(m)]) #Cost function
weights=np.ones(X_train.shape)
while not converged:
    for i in range(m):
        grad0=(theta0+(theta1*X_train[i])-Y_train[i])*weights[i]
        grad1=(theta0+(theta1*X_train[i])-Y_train[i])*weights[i]*X_train[i]
        theta0-=alpha*grad0
        theta1-=alpha*grad1
    e=(0.5/m)*sum([(theta0 + theta1*X_train[i] - Y_train[i])**2 for i in range(m)])
    #print(e)
    if abs(J-e)<0.0001:
        print('Converged')
        converged=True
    J=e
    iterations+=1
    if iterations==10000:
        print('Max. iterations reached')
        converged=True
print('For Stochastic Gradient Descent')
print(theta0,theta1,J,iterations)
plt.scatter(X_train,Y_train,color='blue')
plt.plot([min(X_train),max(X_train)],[theta0+theta1*min(X_train),theta0+theta1*max(X_train)],color='red')
plt.xlabel("Population in 10,000's")
plt.ylabel("Profit in Lakhs(Rs)")
plt.title('Stochastic Gradient Descent')
plt.show()

#Mini Batch Gradient Descent
converged=False
iterations=0
theta0=np.random.random(1)
theta1=np.random.random(1)
batch_size=10
J = (0.5/m)*sum([(theta0 + theta1*X_train[i] - Y_train[i])**2 for i in range(m)]) #Cost function
print(theta0,theta1)
while not converged:
    for i in range((m//batch_size)+1):
        X_batch=np.array(X_train[i*batch_size:(i+1)*batch_size])
        Y_batch=np.array(Y_train[i*batch_size:(i+1)*batch_size])
        grad0=(1/batch_size)*sum(theta0+(theta1*X_batch)-Y_batch)
        grad1=(1/batch_size)*sum((theta0+(theta1*X_batch)-Y_batch)*X_batch)
        theta0-=alpha*grad0
        theta1-=alpha*grad1
    e=(0.5/m)*sum([(theta0 + theta1*X_train[i] - Y_train[i])**2 for i in range(m)])
    #print(e)
    if abs(J-e)<0.0001:
        print('Converged')
        converged=True
    J=e
    iterations+=1
    if iterations==10000:
        print('Max. iterations reached')
        converged=True
print('For Mini Batch Gradient Descent')
print(theta0,theta1,J,iterations)
plt.scatter(X_train,Y_train,color='blue')
plt.plot([min(X_train),max(X_train)],[theta0+theta1*min(X_train),theta0+theta1*max(X_train)],color='red')
plt.xlabel("Population in 10,000's")
plt.ylabel("Profit in Lakhs(Rs)")
plt.title('Mini Batch Gradient Descent')
plt.show()

# Ridge Regression
from sklearn.linear_model import Ridge

RegR = Ridge(alpha=1)
RegR.fit(X_train.reshape(-1,1),Y_train.reshape(-1,1))
prediction = RegR.predict(X_test.reshape(-1,1))

errR = sum((prediction-Y_test)**2)/len(Y_test)
print(err)
print(RegR.intercept_,RegR.coef_)
plt.scatter(X_train,Y_train,color='blue')
plt.plot([min(X_train),max(X_train)],[RegR.intercept_+RegR.coef_[0]*min(X_train),RegR.intercept_+RegR.coef_[0]*max(X_train)],color='red')
plt.xlabel("Population in 10,000's")
plt.ylabel("Profit in Lakhs(Rs)")
plt.title('Ridge Regression')
plt.show()
#plot_pred(RegR.intercept_[0],RegR.coef_[0],data["Population in 10,000's"],data['Profit In Lakhs(Rs)'],'Ridge Regression')

# Elastic Net Regreesion
from sklearn.linear_model import ElasticNet

RegrE = ElasticNet(alpha=1)
RegrE.fit(X_train.reshape(-1,1),Y_train.reshape(-1,1))
prediction = RegrE.predict(X_test.reshape(-1,1))

errE = sum((prediction-Y_test)**2)/len(Y_test)
print(errE)
print(RegrE.intercept_,RegrE.coef_)
plt.scatter(X_train,Y_train,color='blue')
plt.plot([min(X_train),max(X_train)],[RegrE.intercept_+RegrE.coef_[0]*min(X_train),RegrE.intercept_+RegrE.coef_[0]*max(X_train)],color='red')
plt.xlabel("Population in 10,000's")
plt.ylabel("Profit in Lakhs(Rs)")
plt.title('Elastic Net Regression')
plt.show()

# Lasso Regression
from sklearn.linear_model import Lasso

RegL = Lasso(alpha=1)
RegL.fit(X_train.reshape(-1,1),Y_train.reshape(-1,1))
prediction = RegL.predict(X_test.reshape(-1,1))

errL = sum((prediction-Y_test)**2)/len(Y_test)
print(errL)
print(RegL.intercept_,RegL.coef_)
plt.scatter(X_train,Y_train,color='blue')
plt.plot([min(X_train),max(X_train)],[RegL.intercept_+RegL.coef_[0]*min(X_train),RegL.intercept_+RegL.coef_[0]*max(X_train)],color='red')
plt.xlabel("Population in 10,000's")
plt.ylabel("Profit in Lakhs(Rs)")
plt.title('Lasso Regression')
plt.show()