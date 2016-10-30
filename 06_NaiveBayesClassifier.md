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

## Q2. Categorical Feature Predictor [8pts]
The categorical distribution with $l$ categories $\{0,\ldots,l-1\}$ is characterized by parameters $\mathbf{p} = (p_0,\dots,p_{l-1})$:
$$ P(z; \mathbf{p}) = p_0^{[z=0]}p_1^{[z=1]}\ldots p_{l-1}^{[z=l-1]} $$

where $[z=t]$ is 1 if $z$ is $t$ and 0 otherwise.

Given $n$ samples $z_1, \ldots, z_n$ from the above distribution, the smoothed-MLE for each $p_t$ is:
$$ \hat{p_t} = \frac{n_t + \alpha}{n + l\alpha} $$

where $n_t = \sum_{j=1}^{n} [z_j=t]$, i.e., the number of times the label $t$ occurred in the sample. The smoothing is done to avoid zero-count problem (similar in spirit to $n$-gram model in NLP).

```
class CategoricalPredictor:
    """ Feature predictor for a categorical feature.
        Attributes: 
            p (dict) : dictionary of vector containing per class probability of a feature value;
                    the keys of dictionary should exactly match the values taken by this feature
    """
    # feel free to define and use any more attributes, e.g., number of classes, etc
    def __init__(self, x, y, alpha=1) :
        """ initializes the predictor statistics (mu, sigma) for Gaussian distribution
        Inputs:
            x (array_like): feature values (continuous)
            y (array_like): class labels (0,...,k-1)
        """
        categories = x.unique()
        labels = y.unique()
        self.k = len(labels)
        self.p = {}
        occurance = {}
        for each in categories:
            self.p[each] = np.zeros(self.k)
            occurance[each] = [0] * self.k
            
        label_counts = [0] * self.k
        for sex, label in zip(x, y):
            occurance[sex][label] += 1
            label_counts[label] += 1
        
#         print (occurance['Male'][1] + alpha)/ float(label_counts[1] + len(self.p) * alpha)
        for each in categories:
            for j in xrange(self.k):
                self.p[each][j] = (occurance[each][j] + alpha) / float(label_counts[j] + len(self.p) * alpha)
                
    def partial_log_likelihood(self, x):
        """ log likelihood of feature values x according to each class
        Inputs:
            x (array_like): vector of feature values
        Outputs:
            (array_like): matrix of log likelihood for this feature
        """
#         from collections import Counter
#         cnt = Counter(x)
#         most_common = cnt.most_common()[0][0]
        new_x = [0] * len(x)
        for i in xrange(len(x)):
            if self.p.has_key(x[i]):
                new_x[i] = 1
            else:
                new_x[i] = 0
        
        result = np.zeros((len(x),self.k))
        for i in xrange(len(new_x)):
            for j in xrange(self.k):
                result[i,j] = stats.bernoulli.logpmf(new_x[i],self.p[x[i]][j])
        return result
    
# AUTOLAB_IGNORE_START
f = CategoricalPredictor(df['sex'], df['label'])
print f.p
print f.partial_log_likelihood(pd.Series(['Male','Female','Male']))
# print f.partial_log_likelihood(df['work_class'])
# AUTOLAB_IGNORE_STOP
```

## Q3 Putting things together [10pts]
It's time to put all the feature predictors together and do something useful! You will implement two functions in the following class.

1. **__init__()**: Compute the log prior for each class and initialize the feature predictors (based on feature type). The smoothed prior for class $t$ is given by
$$ prior(t) = \frac{n_t + \alpha}{n + k\alpha} $$
where $n_t = \sum_{j=1}^{n} [y_j=t]$, i.e., the number of times the label $t$ occurred in the sample. 

2. **predict()**: For each instance and for each class, compute the sum of log prior and partial log likelihoods for all features. Use it to predict the final class label. Break ties by predicting the class with lower id.

```
class NaiveBayesClassifier:
    """ Naive Bayes classifier for a mixture of continuous and categorical attributes.
        We use GaussianPredictor for continuous attributes and MultinomialPredictor for categorical ones.
        Attributes:
            predictor (dict): model for P(X_i|Y) for each i
            log_prior (array_like): log P(Y)
    """
    # feel free to define and use any more attributes, e.g., number of classes, etc
    def __init__(self, df, alpha=1):
        """initializes predictors for each feature and computes class prior
        Inputs:
            df (pd.DataFrame): processed dataframe, without any missing values.
        """
        y = df['label']
        n = len(y)
        self.k = len(y.unique())
        
        # Calculate log_prior
        self.log_prior = np.zeros(self.k)
        occurance = [0] * len(y)
        for each in df['label']:
            occurance[each] += 1
        for i in xrange(self.k):
            self.log_prior[i] = np.log((occurance[i] + alpha) / float(n + self.k * alpha))
        
        # Get predictors
        columns = df.columns.tolist()
        self.predictor = {}
        for each in columns:
            if each != 'label':
                if df[each].dtype == 'int64':
                    self.predictor[each] = GaussianPredictor(df[each], df['label'])
                else:
                    self.predictor[each] = CategoricalPredictor(df[each], df['label'], alpha)
        
    def predict(self, x):
        """ predicts label for input instances from log_prior and partial_log_likelihood of feature predictors
        Inputs:
            x (pd.DataFrame): processed dataframe, without any missing values and without class label.
        Outputs:
            (array_like): array of predicted class labels (0,..,k-1)
        """
#         labels = x['label']
        if 'label' in x.columns:
            x = x.drop('label',1)
        if 'index' in x.columns:
            x = x.drop('index',1)
        column_names = x.columns
        length = len(x)
        data = np.zeros((length, self.k))
        for each in column_names:
            column_elements = x[each]
            data += self.predictor[each].partial_log_likelihood(column_elements)
        data = data + self.log_prior
#         print data
        output_list = pd.Series(np.random.randn(length))
        count = 0
        for each in data:
            index = np.argmax(each)
            output_list[count] = index
            count += 1
        result = output_list
        return result.as_matrix()
    
# AUTOLAB_IGNORE_START
c = NaiveBayesClassifier(df, 0)
y_pred = c.predict(df)

print y_pred.shape
print y_pred
# AUTOLAB_IGNORE_STOP
```

## Q5. Evaluation - Error rate [2pts]
If a classifier makes $n_e$ errors on a data of size $n$, its error rate is $n_e/n$. Fill the following function, to evaluate your classifier.

```
def evaluate(y_hat, y):
    """ Evaluates classifier predictions
        Inputs:
            y_hat (array_like): output from classifier
            y (array_like): true class label
        Output:
            (double): error rate as defined above
    """
    error = 0
    for i, j in zip(y, y_hat):
        if i != j:
            error += 1
    return error/float(len(y))

# AUTOLAB_IGNORE_START
evaluate(y_pred, df['label'])
# AUTOLAB_IGNORE_STOP
```
0.24892248524633645

Our implementation yields 0.17240236058616804.
https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.norm.html
