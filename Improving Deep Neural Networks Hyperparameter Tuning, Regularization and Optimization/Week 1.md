# Train / Dev / Test sets
- As a machine learning engineer you have to make choices about the hyperparameters , such as ( number of layers, Hidden units , activation functions)
- It is impossible for a ml engineer to figure out the best parameters in th start  so its a iterative process
- The loop goes like this. idea  >> code >> experiment 
- It turns out if a researcher have a experince in NLP and now he wants to do research in CV ,  his all experince can not be apply to the new domain, every domain 
   has its own constaints and solution.
-  Your data will be split into three parts
  - Train data
  - Validation data/ dev data.
  - Test Data
- Tune your hyperparemeters according to result on validation data. once you have good result on validation data then try that algorithm on Test set
- so the trend on the ratio of splitting the models:
  -If size of the dataset is 100 to 1000000 ==> 60/20/20
  - If size of the dataset is 1000000 to INF ==> 98/1/1 or 99.5/0.25/0.25
-The trend now gives the training data the biggest sets.
- The data set of training and validation set  should be from same distribution.
  - For example if cat training pictures is from the web and the dev/test pictures are from users cell phone they will
mismatch. It is better to make sure that dev and test set are from the same distribution.
- It might be ok to not to have a test set.
- Evalutue the different models on dev set

# Bias / Variance

- Bias / variance are easy to learn but difficult to master
- There is a trend  of bias variance trade off.

- If your model is underfitting it has the high bias
- if your model is over fitting it has the high varience
- If your model is balnce then it is perfectly fit

