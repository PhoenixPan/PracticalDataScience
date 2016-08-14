##Data, glorious data. 

Data = Singal + Noise
Goal of the predictor: find the signal, ignore the noise. 
Garbage in = Garbage out

Issues to consider and trade-offs

1. Interpretable;
2. Simple;
3. Accurate;
4. Fast;
5. Scalable (Why Netflex never used the algorithm that won its $ 1M prize).

In sample error (resubstitution error): Errors on the training set that we built with. The data we have.

Out of sample error (generalization error): Errors on the data set that wasn't used to train the predictor. The data we don't have but want to predict. 

Importance: Out > In

Overfitting: Happens when a prediction model is excessively complex. Sometimes, simple rules do better than complicated rules.

Split data into: 

1. Training set: Building model for prediction);
2. Testing set: Test the model, be aware of overfitting. Use testing set for trainging only if you have validation set;
3. Validation set (optional): To validate your model at the end, not used for training  

Apply to the validation set (testing set if it's abscent) for only one time. Otherwise, we are using the test set to train the model. 

