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

![image](https://user-images.githubusercontent.com/36159918/190708112-e35d4c47-fa33-46eb-a586-9b4261e965a7.png)

**High Varince**
- if you have training error 1%
- if you have validation/dev error 11 %
- Its mean that your algorithm is performing well on the training set but the performance  is low on the dev set. The algorithm explicitly learn the trainning data 
  This is the example of the high varinance
  
  **High bias**
 - if  you have training error 15%
 - if you have validation/ dev error 16 %
 - Its mean that the algorithm was not able to do well on training data  and this is the  high bias.
 
  **Both**
 - if you have training error 15 % and
 - if you  have validation/ dev error 30 % . Its mean that the algorithm has the both problems, high bias and high variance.
 **best**
 - if you have 0.5 % training error
 - i fyou have 1 % dev error it has the low bias and low varience
- These Assumptions came from that human has 0% error. If the problem isn't like that you'll need to use human
error as baseline.

# Basic Recipe for Machine Learning

- If your algorithm has a high bias:
   - Try to make your NN bigger (size of hidden units, number of layers) >> (almost work for this problem)
   - Try a different model that is suitable for your data.(Means different architecture )
   - Try to run it longer.(Least thing to do does not help always)
   - Different (advanced) optimization algorithms.(Such as adam , Rsprom etc)
 -If your algorithm has a high variance:
     - More data.
     - Try regularization.
     - Try a different model that is suitable for your data.
  - You should try the previous two points until you have a low bias and low variance.
  - In the older days before deep learning, there was a "Bias/variance tradeoff". But because now you have more
   options/tools for solving the bias and variance problem its really helpful to use deep learning.
  - Training a bigger neural network never hurts (Pretty much always reduce the bias)
  - Training on more data (pretty much always reduce varience)
  - With out hurting other we can deal with bias and vairince now we dont have to worry about trade off


# Regularization
if you have the high varince(over fitting) problem you can try the regularization technique to reduce the varience.
- L1 matrix norm:
   - ||W|| = Sum(|w[i,j]|) # sum of absolute values of all w
-  L2 matrix norm because of arcane technical math reasons is called Frobenius norm:
   - ||W||^2 = Sum(|w[i,j]|^2) # sum of all w squared
   -  Also can be calculated as ||W||^2 = W.T * W if W is a vector
   -  It is usng the euclidean distance that' s why called the l2 regularlization
 - Regularization for logistic regression:
   - The normal cost function that we want to minimize is: J(w,b) = (1/m) * Sum(L(y(i),y'(i)))
   - The L2 regularization version: J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum(|w[i]|^2)
   - You can add the  lamda/2m * b^2 but the most parameters are in w so it would not make a biug difference\
   - The L1 regularization version: J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum(|w[i]|)
   - The L1 regularization version makes a lot of w values become zeros, which makes the model size smaller.
   - L2 regularization is being used much more often.
   - lambda here is the regularization parameter (hyperparameter)
  
  - Regularization for NN:
  - The normal cost function that we want to minimize is:
      J(W1,b1...,WL,bL) = (1/m) * Sum(L(y(i),y'(i)))
  - The L2 regularization version:
    J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum((||W[l]||^2)
  - We stack the matrix as one vector (mn,1) and then we apply sqrt(w1^2 + w2^2.....)
  - To do back propagation (old way):
      dw[l] = (from back propagation)
  - The new way:
      dw[l] = (from back propagation) + lambda/m * w[l]
  - So plugging it in weight update step:
      -  '''
           w[l] = w[l] - learning_rate * dw[l]
           = w[l] - learning_rate * ((from back propagation) + lambda/m * w[l])
            = w[l] - (learning_rate*lambda/m) * w[l] - learning_rate * (from back propagation)
            = (1 - (learning_rate*lambda)/m) * w[l] - learning_rate * (from back propagation)
         '''
  -  In practice this penalizes large weights and effectively limits the freedom in your model.
  -  The new term (1 - (learning_rate*lambda)/m) * w[l] causes the weight to decay in proportion to its size.
  -  L2 often called as **weight decay** 


 # Why regularization reduces overfitting?
 Here are some Here are some intuitions:
 
 ![Reg](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Regularliazation.PNG)
 
 Consider the above image for intuitions:
   - if lambda is too large - a lot of w's will be close to zeros which will make the NN simpler (you can think of it as it
      would behave closer to logistic regression).
      - The neourn weights will be small enough that the value would not matters
      - If lambda is good enough it will just reduce some weights that makes the neural network overfit.
   - Intuition 2 (with tanh activation function):
      - If lambda is too large, w's will be small (close to zero) - will use the linear part of the tanh activation function, so we
         will go from non linear activation to roughly linear which would make the NN a roughly linear classifier.
      - If lambda good enough it will just make some of tanh activations roughly linear which will prevent overfitting.
   - Implementation tip: if you implement gradient descent, one of the steps to debug gradient descent is to plot the cost
   function J as a function of the number of iterations of gradient descent and you want to see that the cost function J
   decreases monotonically after every elevation of gradient descent with regularization. If you plot the old definition of J (no
   regularization) then you might not see it decrease monotonically.

  
j
