import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import statistics

# Defining functions for calculating statistic measures
def mean(l):
    return sum(l)/len(l)

def median(l):
    p=len(l)
    l=np.sort(l)
    if p%2!=0:
        return l[p//2]
    else:
        return (l[p//2]+l[(p//2)-1])/2

def variance(l):
    p=len(l)
    m=mean(l)
    k=0
    for i in l:
        k+=i*i
    k=k//p
    k-=m*m
    return k

def MAD(l):
    k=median(l)
    p=[]
    for i in l:
        p+=[abs(i-k)]
    return median(p)

# Reading excel files
data1=pd.read_excel("data1.xlsx")
data2=pd.read_excel("data3.xlsx")

data1_x=data1['x']
data1_y=data1['y']
data2_x=data2['x']
data2_y=data2['y']

#PLotting various plots
figure,ax=plt.subplots(2,4)

ax[0,0].scatter(data1_x,data1_y,color='blue')
ax[0,1].hist2d(data1_x,data1_y)
ax[0,2].pcolormesh([data1_x,data1_y])
ax[0,3].boxplot([data1_x,data1_y])

ax[1,0].scatter(data2_x,data2_y,color='red')
ax[1,1].hist2d(data2_x,data2_y)
ax[1,2].pcolormesh([data2_x,data2_y])
ax[1,3].boxplot([data2_x,data2_y])

plt.show()

mean1=[mean(data1_x),mean(data1_y)]
mean2=[mean(data2_x),mean(data2_y)]

median1=[median(data1_x),median(data1_y)]
median2=[median(data2_x),median(data2_y)]

variance1=[variance(data1_x),variance(data1_y)]
variance2=[variance(data2_x),variance(data2_y)]

std1=[variance1[0]**0.5,variance1[1]**0.5]
std2=[variance2[0]**0.5,variance2[1]**0.5]

mad2=[MAD(data2_x),MAD(data2_y)]

print('Mean of data1.xlsx',mean1)
print('Mean of data3.xlsx',mean2)

print('Median of data1.xlsx',median1)
print('Median of data3.xlsx',median2)

print('Variance of data1.xlsx',variance1)
print('Variance of data3.xlsx',variance2)

#print(std1)
#print(std2)

#print(mad2)

#Outlier detection
outlier_x_std=[]
outlier_y_std=[]
outlier_x_mad=[]
outlier_y_mad=[]

for i in range(len(data2_x)):
    if abs(data2_x[i]-mean2[0])>3*std2[0] or abs(data2_y[i]-mean2[1])>3*std2[1]:
        outlier_x_std+=[data2_x[i]]
        outlier_y_std+=[data2_y[i]]
    if abs(data2_x[i]-median2[0])>3*mad2[0] or abs(data2_y[i]-median2[1])>3*mad2[1]:
        outlier_x_mad+=[data2_x[i]]
        outlier_y_mad+=[data2_y[i]]

figure,out=plt.subplots(2,1)

out[0].scatter(data2_x,data2_y,color='blue')
out[0].scatter(outlier_x_std,outlier_y_std,color='red')
#out[0].legend('Data','Outlier')
out[0].set_title('standard deviation approach')
out[0].set_xlabel('x')
out[0].set_ylabel('y')

out[1].scatter(data2_x,data2_y,color='blue')
out[1].scatter(outlier_x_mad,outlier_y_mad,color='red')
#out[1].legend('Data','Outlier')
out[1].set_title('MAD approach')
out[1].set_xlabel('x')
out[1].set_ylabel('y')

plt.show()