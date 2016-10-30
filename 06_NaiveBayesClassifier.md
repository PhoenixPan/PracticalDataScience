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



https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.norm.html
