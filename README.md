# Optimal_and_Naive_Bayes_Classifiers
Implementation of optimal bayes and naïve bayes methods on the Iris dataset, both from scratch and by using scikit-learn.

<h2>Naive Bayes from scratch:</h2>

First, I seperate the data into 3 different classes: Iris-setosa, Iris-versicolor, and Iris-virginica.

Next, I define a function which calculates the **mean** and the **variance** of these 3 classes. The function in the code above is as below:

```ruby
def mean_var_calc(class_i)
```

Next I implement function which generates a a **Gaussian distribution** given the mean and the variance, so the distribution of each class is calculated as follows:

```ruby
def guassian_distribution (x, mean_class, var_class)
def probabilty_calc (x, class_i, rows)
```

The decision rule is implemented by comparing the probability of belonging to each class as below:

```ruby
if (p_setosa >= p_versicolor and p_setosa >= p_virginica):
        return ('setosa')
    elif (p_versicolor >= p_setosa and p_versicolor >= p_virginica):
        return('versicolor')
    elif (p_virginica >= p_setosa and p_virginica >= p_versicolor):
        return ('virginica')
```

Calculating the **Confusion Matrix** and the **accuracy** of the model, the results are as follows:

confusion matrix = 
[[17, 1, 1], [0, 10, 1], [0, 1, 15]]

acuuracy =  0.9545454545454546

<h2>Optimal Bayes from scratch:</h2>

Since there is no independency here, I defined a function to calculate the **mean** and the **covariance** between all the classes as below:

```ruby
def mean_cov_calc(class_i)
```

Next, I implemented a Gx_calculator function, which calculates the weights of the **Multivariate Gaussian Distribution**:

```ruby
def Gx_calc (X, rows, class_i)
```

The decision making rule is the same as before. The **Confusion Matrix** and the **accuracy** of the model, the results are as follows:

confusion matrix = 
[[13, 0, 2], [0, 18, 2], [0, 0, 10]]

acuuracy =  0.9534883720930233

<h2>Naive Bayes with sickit-learn:</h2>

I implemented the model using the sklearn library as below:

```ruby
from sklearn.naive_bayes import GaussianNB
```

By fitting the model on the **train** data, and later testing it on the **test set**, the results are as follows:

confusion matrix =  [[11  0  0]
 [ 0 13  0]
 [ 0  1  5]]
 
accuracy =  0.9666666666666667
