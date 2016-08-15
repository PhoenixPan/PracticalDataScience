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

Receiver Operating Charactistic(ROC) curves  
X axis: P(FP) = 1 - speciality    
Y axis: P(TP) = sensitivity  

Area under the curve(AUC): The larger the bottom area, the better the prediction is.  
AUC = 0.5: random guessing  
AUC = 1:   perfect classifer  
In general, the AUC above 0.8 considered "good".  





