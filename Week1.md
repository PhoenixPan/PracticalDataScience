##Data, glorious data. 
Basic:

1. Data = Singal + Noise
2. Goal of our model: find the signal, ignore the noise. 
3. Garbage in = Garbage out

Issues to consider and trade-offs:

1. Interpretable;
2. Simple;
3. Accurate;
4. Fast;
5. Scalable (Why Netflex never used the algorithm that won its $ 1M prize).

In sample error (resubstitution error): Errors on the training set that we built with. The data we have.

Out of sample error (generalization error): Errors on the data set that wasn't used to train the predictor. The data we don't have but want to predict. 

Importance: Out > In

Overfitting: Happens when a prediction model is excessively complex. Sometimes, simple rules do better than complicated rules.

Divide your dataset into:

1. Training set: Building model for prediction. This is the only part that should be visable to you when you build the model;
2. Testing set: Test the model, be aware of overfitting. Use testing set for trainging only if you have validation set;
3. Validation set (optional): To validate your model at the end, not used for training  

Keep in mind: 

1. Apply to the validation set (testing set if it's abscent) for only one time. Otherwise, we are using the test set to train the model. 
2. Avoid small sample sizes, otherwise you may get your accuracy by coincidence (flip a coin).
3. Sample your data in time chunks (backtesting)
4. Random assignment or balancing by features (tricky)

Rules of thumb:

1. Large dataset: 60% for training, 20% for testing, 20% for validation 
2. Medium dataset: 60% for training, 40% for testing (no refinery)
3. Small dataset: are you sure you want to do this? 
