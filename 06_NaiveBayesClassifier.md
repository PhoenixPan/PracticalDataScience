# Naive Bayes Classifier

## Introduction
Naive Bayes is a class of simple classifiers based on the Bayes' Rule and strong (or naive) independence assumptions between features. In this problem, you will implement a Naive Bayes Classifier for the Census Income Data Set from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/) (which is a good website to browse through for datasets).

## Dataset Description
The dataset consists 32561 instances, each representing an individual. The goal is to predict whether a person makes over 50K a year based on the values of 14 features. The features, extracted from the 1994 Census database, are a mix of continuous and discrete attributes. These are enumerated below:

#### Continuous (real-valued) features
- age
- final_weight (computed from a number of attributes outside of this dataset; people with similar demographic attributes have similar values)
- education_num
- capital_gain
- capital_loss
- hours_per_week

#### Categorical (discrete) features 
- work_class: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
- marital_status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
- sex: Female, Male
- native_country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands  

## Q1. Input preparation [2 pts]
First, you need to load in the above data, provided to you as a CSV file. As the data is from UCI repository, it is already quite clean. However, some instances contain missing values (represented as ? in the CSV file) and these have to be discarded from the training set. Also, replace the `income` column with `label`, which is 1 if `income` is `>50K` and 0 otherwise.

```
import pandas as pd
import numpy as np
import scipy
from scipy import stats
```
```
def load_data(file_name):
    """ loads and processes data in the manner specified above
    Inputs:
        file_name (str): path to csv file containing data
    Outputs:
        pd.DataFrame: processed dataframe
    """
    df = pd.read_csv(file_name, na_values='?')
    df = df.dropna()
    df = df.rename(columns=lambda x: x.replace('income', 'label'))
    for index, row in df.iterrows():
        if row['label'] == '>50K':
            df.set_value(index, 'label', '1')
        else:
            df.set_value(index, 'label', '0')
    df['label'] = df['label'].apply(pd.to_numeric) 
    df = df.reset_index()
    return df

# AUTOLAB_IGNORE_START
df = load_data('census.csv')
print df.isnull().values.any()  # True
print df.tail()
print len(df)
# AUTOLAB_IGNORE_STOP
```

## Overview of Naive Bayes classifier
Let $X_1, X_2, \ldots, X_k$ be the $k$ features of a dataset, with class label given by the variable $y$. A probabilistic classifier assigns the most probable class to each instance $(x_1,\ldots,x_k)$, as expressed by
$$ \hat{y} = \arg\max_y P(y\ |\ x_1,\ldots,x_k) $$

Using Bayes' theorem, the above *posterior probability* can be rewritten as
$$ P(y\ |\ x_1,\ldots,x_k) = \frac{P(y) P(x_1,\ldots,x_n\ |\ y)}{P(x_1,\ldots,x_k)} $$
where
- $P(y)$ is the prior probability of the class
- $P(x_1,\ldots,x_k\ |\ y)$ is the likelihood of data under a class
- $P(x_1,\ldots,x_k)$ is the evidence for data

Naive Bayes classifiers assume that the feature values are conditionally independent given the class label, that is,
$ P(x_1,\ldots,x_n\ |\ y) = \prod_{i=1}^{k}P(x_i\ |\ y) $. This strong assumption helps simplify the expression for posterior probability to
$$ P(y\ |\ x_1,\ldots,x_k) = \frac{P(y) \prod_{i=1}^{k}P(x_i\ |\ y)}{P(x_1,\ldots,x_k)} $$

For a given input $(x_1,\ldots,x_k)$, $P(x_1,\ldots,x_k)$ is constant. Hence, we can simplify omit the denominator replace the equality sign with proportionality as follows:
$$ P(y\ |\ x_1,\ldots,x_k) \propto P(y) \prod_{i=1}^{k}P(x_i\ |\ y) $$

Thus, the class of a new instance can be predicted as $\hat{y} = \arg\max_y P(y) \prod_{i=1}^{k}P(x_i\ |\ y)$. Here, $P(y)$ is commonly known as the **class prior** and $P(x_i\ |\ y)$ termed **feature predictor**. The rest of the assignment deals with how each of these $k+1$ probability distributions -- $P(y), P(x_1\ |\ y), \ldots, P(x_k\ |\ y)$ -- are estimated from data.


**Note**: Observe that the computation of the final expression above involve multiplication of $k+1$ probability values (which can be really low). This can lead to an underflow of numerical precision. So, it is a good practice to use a log transform of the probabilities to avoid this underflow.

** TL;DR ** Your final take away from this cell is the following expression:
$$\hat{y} = \arg\max_y \underbrace{\log P(y)}_{log-prior} + \underbrace{\sum_{i=1}^{k} \log P(x_i\ |\ y)}_{log-likelihood}$$

## Feature Predictor
The beauty of a Naive Bayes classifier lies in the fact we can mix-and-match different likelihood models for each feature predictor according to the prior knowledge we have about it and these models can be varied independent of each other. For example, we might know that $P(X_i|y)$ for some continuous feature $X_i$ is normally distributed or that $P(X_i|y)$ for some categorical feature follows multinomial distribution. In such cases, we can directly plugin the pdf/pmf of these distributions in place of $P(x_i\ |\ y)$.

In this assignment, you will be using two classes of likelihood models:
- Gaussian model, for continuous real-valued features (parameterized by mean $\mu$ and variance $\sigma$)
- Categorical model, for discrete features (parameterized by $\mathbf{p} = <p_0,\ldots,p_{l-1}>$, where $l$ is the number of values taken by this categorical feature)

You need to implement a predictor class for each likelihood model. Each predictor should implement two functionalities:
- **Parameter estimation `init()`**: Learn parameters of the likelihood model using MLE (Maximum Likelihood Estimator). You need to keep track of $k$ sets of parameters, one for each class.
- **Partial Log-Likelihood computation for *this* feature `partial_log_likelihood()`**: Use the learnt parameters to compute the probability (density/mass for continuous/categorical features) of a given feature value.

The parameter estimation is for the conditional distributions $P(X|Y)$. Thus, while estimating parameters for a specific class (say class 0), you will use only those data points in the training set (or rows in the input data frame) which have class label 0.


## Q2. Gaussian Feature Predictor [8pts]
The Guassian distribution is characterized by two parameters - mean $\mu$ and standard deviation $\sigma$:
$$ f_Z(z) = \frac{1}{\sqrt{2\pi}\sigma} \exp{(-\frac{(z-\mu)^2}{2\sigma^2})} $$

Given $n$ samples $z_1, \ldots, z_n$ from the above distribution, the MLE for mean and standard deviation are:
$$ \hat{\mu} = \frac{1}{n} \sum_{j=1}^{n} z_j $$

$$ \hat{\sigma} = \frac{1}{n} \sum_{j=1}^{n} (z_j-\hat{\mu})^2 $$

`scipy.stats.norm` would be helpful.

```
class GaussianPredictor:
    """ Feature predictor for a normally distributed real-valued, continuous feature.
        Attributes: 
            mu (array_like) : vector containing per class mean of the feature
            sigma (array_like): vector containing per class std. deviation of the feature
    """
    # feel free to define and use any more attributes, e.g., number of classes, etc
    def __init__(self, x, y) :
        """ initializes the predictor statistics (mu, sigma) for Gaussian distribution
        Inputs:
            x (array_like): feature values (continuous)
            y (array_like): class labels (0,...,k-1)
        """
        self.k = len(y.unique())
        self.mu = np.zeros(self.k)
        self.sigma = np.zeros(self.k)
        
        class_data = []
        for i in xrange(self.k):
            class_data.append(list())
        
        labels = y.unique().tolist()
        for age, label in zip(x, y):
            class_data[label].append(age)
        
        for i in xrange(self.k):
            self.mu[i] = np.average(class_data[i])
            self.sigma[i] = np.std(class_data[i])
        
    def partial_log_likelihood(self, x):
        """ log likelihood of feature values x according to each class
        Inputs:
            x (array_like): vector of feature values
        Outputs:
            (array_like): matrix of log likelihood for this feature alone
        """
        result = np.zeros((len(x),self.k))
        for i in xrange(len(x)):
            for j in xrange(self.k):
                result[i,j] = stats.norm.logpdf(x[i],loc=self.mu[j],scale=self.sigma[j])
        return result

# AUTOLAB_IGNORE_START
f = GaussianPredictor(df['age'], df['label'])
print f.mu
print f.sigma
print f.partial_log_likelihood(pd.Series([43,40,100,10]))
# print f.partial_log_likelihood(df['age'])
# AUTOLAB_IGNORE_STOP
```

https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.norm.html
