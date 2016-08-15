##Data, glorious data. 
Basic:  
1. Data = Singal + Nois;  
2. Goal of our model: find the signal, ignore the noise;  
3. Garbage in = Garbage out.

Issues to consider and trade-offs:  
1. Interpretable;  
2. Simple;  
3. Accurate;  
4. Fast;  
5. Scalable (Why Netflex never used the algorithm that won its $ 1M prize).

Divide your dataset into:  
1. Training set: Building model for prediction. This is the only part that should be visable to you when you build the model;  
2. Testing set: Test the model, be aware of overfitting. Use testing set for trainging only if you have validation set;  
3. Validation set (optional): To validate your model at the end, not used for training.  

Keep in mind:  
1. Apply to the validation set (testing set if it's abscent) for only one time. Otherwise, we are using the test set to train the model;  
2. Avoid small sample sizes, otherwise you may get your accuracy by coincidence (flip a coin);  
3. Sample your data in time chunks (backtesting);  
4. Random assignment or balancing by features (tricky).  

Overfitting: Happens when a prediction model is excessively complex. Sometimes, simple rules do better than complicated rules.

Rules of thumb:  
1. Large dataset: 60% for training, 20% for testing, 20% for validation;  
2. Medium dataset: 60% for training, 40% for testing (no refinery);  
3. Small dataset: are you sure you want to do this?  


## Types of error

Out of sample error > In sample error:  
1. In sample error (resubstitution error): Errors on the training set that we built with. The data we have;  
2. Out of sample error (generalization error): Errors on the data set that wasn't used to train the predictor. The data we don't have but want to predict. 

Positive = identified  
Negative = rejected

True Positive  
False positive  
True Negative  
False Negative

Key fractions:  
Sensitivity = TP / (TP + FN) --- accuracy of all the trues  
Specificity = TN / (TN + FP) --- false of all the falses  
Positive Predictive Value = TP / (TP + FP) --- accuracy of all the positives    
Negative Predictive Value = TN / (FN + TN) --- accuracy of all the negatives   
Accuracy outcome = TP + FN/ ALL SUM  

Exercise: Assume that some disease has a 0.1% prevalence in the population. If a test kit gives 99% sensitivity and 99% speciality, what is the possibility of a person, given the test is positive, really has the disease?  
```  
              Disease
           +          -
Test +    99         999
     -     1       98901
```
Positive Predictive Value = 99/(99+999) ≈ 9%  
We need more accurate prediction when the event has lower change to happen to ensure the same accuracy.

Error measures for continuous data:  
1. Mean squard error (MSE): 1/n * ∑(Prediction - Truth)^2 (similar to standard deviation, low tolerance for outliers)  
2. Root mean squard error (RMSE): √MSE  (bring back the scale which has been extended by square in MSE, more robust)  
3. Sensitivity: few missed positives
4. Speciality: few negatives called positives
5. Accuracy:  weights false positive/negative equally
6. Concodance: example: kappa

##Receiver Operating Charactistic(ROC) curves  
X axis: P(FP) = 1 - speciality    
Y axis: P(TP) = sensitivity  

Area under the curve(AUC): The larger the bottom area, the better the prediction is.  
AUC = 0.5: random guessing  
AUC = 1:   perfect classifer  
In general, the AUC above 0.8 considered "good".  

##Cross Validation  
1. Accuracy on training set (resubstitution accuracy) is optimistic;  
2. A better estimate comes from an __independent__ set (test set accuracy);  
3. Don't use test set when building model or use it as a part of training set;  
4. So we estimate the test set accuracy with the training set.  

Approach:  
1. Use the training set;  
2. Split it into training/test sets;    
3. Build a model on the training set;  
4. Evaluate on the test set;  
5. Repeat and average the estimated errors.  
(Since we split the original dataset, it can still provides an objective result when we test it at the end)  
  
Used for:  
1. Picking variables to introduce in a model;  
2. Picking the type of prediction function to use;  
3. Picking the parameters in the prediction function;  
4. Corporating different predictors.  
  
Model building:  
Random subsampling: randomly choose training/test sets sectors;  
K-Fold: example of 3-fold, we create three combination on one dataset: train/train/test, train/test/train, test/train/train;    
Leave one out: Leave a certain width as test set, put it in many places. 10000,01000,...;  
  
Consideration:  
1. For time series data, data must be used in "chunks";  
2. For K-fold cross validation, larger k means less bias, more variance, smaller k means more bias, less variance;  
3. Random sampling must be done without replacement, otherwise you get bootstrap and underestimate the error;  
4. Estimate errors in independent dataset.  
  

##Most common mistake: Unrelated data  
Example: Chocolate consumption and Nobel Prize are not really related, as Europeans countries may happen to like chocolate more and they are also in charge of issuing the prize. 







