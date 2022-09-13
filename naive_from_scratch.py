#!/usr/bin/env python
# coding: utf-8

# In[19]:


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
                


# In[20]:


def mean_var_calc(class_i):
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

    var_vector = [var_sl, var_sw, var_pl, var_pw]

    return mean_vector, var_vector


# In[21]:


import math
def guassian_distribution (x, mean_class, var_class):
    fx = (1 / ( (2 * math.pi * var_class) ** 0.5 )) * math.exp(-((x - mean_class) ** 2) / (2 * var_class))
    return fx


# In[22]:


def probabilty_calc (x, class_i, rows):
    mean_class, var_class = mean_var_calc(class_i)
    p_sl = guassian_distribution (x[0], mean_class[0], var_class[0])
    p_sw = guassian_distribution (x[1], mean_class[1], var_class[1])
    p_pl = guassian_distribution (x[2], mean_class[2], var_class[2])
    p_pw = guassian_distribution (x[3], mean_class[3], var_class[3])
    py = len(class_i) / len(rows)
    probability = py * p_sl * p_sw * p_pl * p_pw 
    return probability


# In[23]:


def decision_rule (x, setosa, versicolor, virginica, rows):
    p_setosa = probabilty_calc (x, setosa, rows)
    p_versicolor = probabilty_calc (x, versicolor, rows)
    p_virginica = probabilty_calc (x, virginica, rows)

    if (p_setosa >= p_versicolor and p_setosa >= p_virginica):
        return ('setosa')
    elif (p_versicolor >= p_setosa and p_versicolor >= p_virginica):
        return('versicolor')
    elif (p_virginica >= p_setosa and p_virginica >= p_versicolor):
        return ('virginica')


# In[24]:


T1 = F1 = T2 = F2 = T3 = F3 = 0 
for i in range (len(test)):
    estmiation = decision_rule(test[i], setosa, versicolor, virginica, train)
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

