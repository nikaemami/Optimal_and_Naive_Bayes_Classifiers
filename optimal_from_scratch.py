#!/usr/bin/env python
# coding: utf-8

# In[21]:


import csv
import random
file = open("/Users/Nika/Desktop/Iris.csv")
csvreader = csv.reader(file)
header = next(csvreader)

rows = []
for row in csvreader:
        rows.append(row)

for i in range (len(rows)):
    for j in range (4):
        rows[i][j] = float (rows[i][j])


train=[]
n = int(0.7 * len(rows))
train = random.sample(rows, n)

test=[]
for i in range (len(rows)):
        if (rows[i] in train):
                continue
        else:
                test.append(rows[i])

setosa=[]
versicolor=[]
virginica=[]
for i in range (len(train)):
        if(train[i][4] == 'Iris-setosa'):
                setosa.append(train[i])
        if(train[i][4] == 'Iris-versicolor'):
                versicolor.append(train[i])
        if(train[i][4] == 'Iris-virginica'):
                virginica.append(train[i])
                


# In[22]:


def mean_cov_calc(class_i):
    sum_sepal_length = sum_sepal_width = sum_petal_length = sum_petal_width = 0
    for i in range (len(class_i)):
        sum_sepal_length += float(class_i[i][0])
        sum_sepal_width += float(class_i[i][1])
        sum_petal_length += float(class_i[i][2])
        sum_petal_width += float(class_i[i][3])
    
    mean_sepal_length = sum_sepal_length / len(class_i)
    mean_sepal_width = sum_sepal_width / len(class_i)
    mean_petal_length = sum_petal_length / len(class_i)
    mean_petal_width = sum_petal_width / len(class_i)

    mean_vector = [mean_sepal_length, mean_sepal_width, mean_petal_length, mean_petal_width]

    sigma_sl = sigma_sw = sigma_pl = sigma_pw = 0
    for i in range (len(class_i)):
        sigma_sl += (float(class_i[i][0]) - mean_sepal_length) ** 2
        sigma_sw += (float (class_i[i][1]) - mean_sepal_width) ** 2
        sigma_pl += (float (class_i[i][2]) - mean_petal_length) ** 2
        sigma_pw += (float (class_i[i][3]) - mean_petal_width) ** 2

    var_sl = sigma_sl / (len(class_i))
    var_sw = sigma_sw / (len(class_i))
    var_pl = sigma_pl / (len(class_i))
    var_pw = sigma_pw / (len(class_i))

    cov_sl_pl = cov_sl_sw = cov_sl_pw = cov_sw_pl = cov_sw_pw = cov_pl_pw = 0

    for i in range (len(class_i)):
            cov_sl_pl +=  (float (class_i[i][0]) - mean_sepal_length)*(float (class_i[i][2])  - mean_petal_length)
            cov_sl_sw +=  (float (class_i[i][0])  - mean_sepal_length)*(float (class_i[i][1]) - mean_sepal_width)
            cov_sl_pw +=  (float (class_i[i][0])  - mean_sepal_length)*(float (class_i[i][3])  - mean_petal_width)
            cov_sw_pl += (float (class_i[i][1])  - mean_sepal_width)*(float (class_i[i][2])  - mean_petal_length)
            cov_sw_pw += (float (class_i[i][1])  - mean_sepal_width)*(float (class_i[i][3])  - mean_petal_width)
            cov_pl_pw += (float (class_i[i][2])  - mean_petal_length)*(float (class_i[i][3])  - mean_petal_width)

    cov_sl_pl /= (len(class_i))
    cov_sl_sw /= (len(class_i))
    cov_sl_pw /= (len(class_i))
    cov_sw_pl /= (len(class_i))
    cov_sw_pw /= (len(class_i))
    cov_pl_pw /= (len(class_i))

    cov_matrix = [[var_sl, cov_sl_sw, cov_sl_pl, cov_sl_pw],
                 [cov_sl_sw, var_sw, cov_sw_pl, cov_sw_pw],
                 [cov_sl_pl, cov_sw_pl, var_pl , cov_pl_pw],
                 [cov_sl_pw, cov_sw_pw , cov_pl_pw, var_pw]]

    return mean_vector, cov_matrix


# In[23]:


import numpy
def Gx_calc (X, rows, class_i):
    X = (X[0:4])
    mean_class_i, cov_class_i = mean_cov_calc (class_i)
    W = numpy.dot(-0.5, numpy.linalg.inv(cov_class_i))
    w = numpy.dot(numpy.linalg.inv(cov_class_i),mean_class_i)
    A = numpy.dot(- 0.5 ,numpy.transpose(mean_class_i))
    B = numpy.dot(A, numpy.linalg.inv(cov_class_i))
    w0 = (numpy.dot(B, mean_class_i)) - 0.5 * numpy.log(numpy.linalg.det(cov_class_i)) + numpy.log(len(class_i) / len(rows))
    C = numpy.dot(numpy.transpose(X), W)
    return (numpy.dot(C, X) + numpy.dot(numpy.transpose(w), X)+ w0)


# In[24]:


def decision_make (X, train, setosa, versicolor, virginica):
    G_setosa = Gx_calc (X, train, setosa)
    G_versicolor = Gx_calc (X, train, versicolor)
    G_virginica = Gx_calc (X, train, virginica)
    if (G_setosa >= G_versicolor and G_setosa >= G_virginica):
        return ('setosa')
    elif (G_versicolor >= G_setosa and G_versicolor >= G_virginica):
        return('versicolor')
    elif (G_virginica >= G_setosa and G_virginica >= G_versicolor):
        return ('virginica')


# In[25]:


T1 = F1 = T2 = F2 = T3 = F3 = 0 
for i in range (len(test)):
    estmiation = decision_make (test[i], train, setosa, versicolor, virginica)
    if (estmiation == 'setosa' and test[i][4] == 'Iris-setosa'):
        T1+=1
    elif (estmiation == 'setosa' and test[i][4] != 'Iris-setosa'):
        F1+=1
    elif (estmiation == 'versicolor' and test[i][4] == 'Iris-versicolor'):
        T2+=1
    elif (estmiation == 'versicolor' and test[i][4] != 'Iris-versicolor'):
        F2+=1
    elif (estmiation == 'virginica' and test[i][4] == 'Iris-virginica'):
        T3+=1
    elif (estmiation == 'virginica' and test[i][4] != 'Iris-virginica'):
        F3+=1

print ("confusion matrix = ")
print([[T1, F2, F3], [F1, T2, F3], [F1, F2, T3]])
print("acuuracy = ", (T1 + T2 + T3) / (T1 + F1 + T2 + F2 + T3 + F3))


        

