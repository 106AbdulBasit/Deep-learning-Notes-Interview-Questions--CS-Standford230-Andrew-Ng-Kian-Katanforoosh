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

 
# Dropout Regularization

![Dp](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Drop%20Out.PNG)

- In most cases Andrew Ng tells that he uses the L2 regularization.
- The dropout regularization eliminates some neurons/weights on each iteration based on a probability.
- Code for Inverted dropout:
   '''
   keep_prob = 0.8 # 0 <= keep_prob <= 1
   l = 3 # this code is only for layer 3
   # the generated number that are less than 0.8 will be dropped. 80% stay, 20% dropped
   d3 = np.random.rand(a[l].shape[0], a[l].shape[1]) < keep_prob
   a3 = np.multiply(a3,d3) # keep only the values in d3
   # increase a3 to not reduce the expected value of output
   # (ensures that the expected value of a3 remains the same) - to solve the scaling problem
   a3 = a3 / keep_prob
   '''
   
- Vector d[l] is used for forward and back propagation and is the same for them, but it is different for each iteration (pass)
or training example.
- At test time we don't use dropout. If you implement dropout at test time - it would add noise to predictions.

# Understanding Dropout
- In the previous video, the intuition was that dropout randomly knocks out units in your network. So it's as if on every
iteration you're working with a smaller NN, and so using a smaller NN seems like it should have a regularizing effect.
- Another intuition: can't rely on any one feature, so have to spread out weights.
- It's possible to show that dropout has a similar effect to L2 regularization.
- Dropout can have different keep_prob per layer.
- The input layer dropout has to be near 1 (or 1 - no dropout) because you don't want to eliminate a lot of features.
- you're more worried about some layers overfitting than others, you can set a lower keep_prob for some layers than
others. The downside is, this gives you even more hyperparameters to search for using cross-validation. One other
alternative might be to have some layers where you apply dropout and some layers where you don't apply dropout and
then just have one hyperparameter, which is a keep_prob for the layers for which you do apply dropouts.

- A lot of researchers are using dropout with Computer Vision (CV) because they have a very big input size and almost
never have enough data, so overfitting is the usual problem. And dropout is a regularization technique to prevent
overfitting.
- downside of dropout is that the cost function J is not well defined and it will be hard to debug (plot J by iteration).
   - To solve that you'll need to turn off dropout, set all the keep_prob s to 1, and then run the code and check that it 
    monotonically decreases J and then turn on the dropouts again.

# Other regularization methods

- **Data Augmentation**
- If you  model is over fiiting and it requires more data to solve the problem ,Rather the collecting more data you can augmentain the data.
- For example in Computer Vision:
   - You can flip all your pictures horizontally this will give you m more data instances.
   - You could also apply a random position and rotation to an image to get more data.
- For example in OCR, you can impose random rotations and distortions to digits/letters.
- New data obtained using this technique isn't as good as the real independent data, but still can be used as a
   regularization technique
   
- **Early stopping**
   - In this technique we plot the training set and the dev set cost together for each iteration. At some iteration the dev
      set cost will stop decreasing and will start increasing.
   - We will pick the point at which the training set error and dev set error are best (lowest training cost with lowest dev
      cost).
   -  We will take these parameters as the best parameters.
   -  ![image](https://user-images.githubusercontent.com/36159918/190858101-466ba582-4746-4688-becd-492a8e5ccab1.png)
   -  Andrew prefers to use L2 regularization instead of early stopping because this technique simultaneously tries to
      minimize the cost function and not to overfit which contradicts the **orthogonalization approach** (will be discussed
       further).
   - But its advantage is that you don't need to search a hyperparameter like in other regularization approaches (like
      lambda in L2 regularization).
      
- Model Ensembles:
- Algorithm:
   - Train multiple independent models
   - At test time average their results.
- It can get you extra 2% performance.
- It reduces the generalization error.


# Normalizing inputs

- If you normalize your inputs this will speed up the training process a lot.
- Normalization are going on these steps:
   - Get the mean of the training set: mean = (1/m) * sum(x(i))
   - Subtract the mean from each input: X = X - mean >> (This makes your inputs centered around 0.)
   - Get the variance of the training set: variance = (1/m) * sum(x(i)^2)
   - Normalize the variance. X /= variance
- Use the value of mean and varince of the train set  to normalize the test/dev set , beacuse you dont want to normalize differently

**Why normalize?**

![nr](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Normalizing%20Input.PNG)

- If we don't normalize the inputs our cost function will be deep and its shape will be inconsistent (elongated) then
optimizing it will take a long time.(see the left side of the image)
- But if we normalize it the opposite will occur. The shape of the cost function will be consistent (look more
symmetric like circle in 2D example) and we can use a larger learning rate alpha - the optimization will be faster.
See the right side of the image.

# Vanishing / Exploding gradients

- The Vanishing / Exploding gradients occurs when your derivatives become very small or very big

![gd](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Vanishing.PNG)
- To understand the problem, suppose that we have a deep neural network with number of layers L, and all the activation
functions are linear and each b = 0

   - Then:
      '''
      Y' = W[L]W[L-1].....W[2]W[1]X
      '''
   - Then, if we have 2 hidden units per layer and x1 = x2 = 1, we result in:
      '''
      if W[l] = [1.5 0]
      [0 1.5] (l != L because of different dimensions in the output layer)
      Y' = W[L] [1.5 0]^(L-1) X = 1.5^L # which will be very large
      [0 1.5]
      
      '''
      
      '''
      if W[l] = [0.5 0]
      [0 0.5]
      Y' = W[L] [0.5 0]^(L-1) X = 0.5^L # which will be very small
      [0 0.5]
      '''

- The last example explains that the activations (and similarly derivatives) will be decreased/increased exponentially as a
   function of number of layers.
- So If W > I (Identity matrix) the activation and gradients will explode.
- And If W < I (Identity matrix) the activation and gradients will vanish.

- Recently Microsoft trained 152 layers (ResNet)! which is a really big number. With such a deep neural network, if your
activations or gradients increase or decrease exponentially as a function of L, then these values could get really big or
really small. And this makes training difficult, especially if your gradients are exponentially smaller than L, then gradient
descent will take tiny little steps. It will take a long time for gradient descent to learn anything.
   
   
# Weight Initialization for Deep Networks
- A partial solution to the Vanishing / Exploding gradients in NN is better or more careful choice of the random
   initialization of weights
- In a single neuron (Perceptron model): Z = w1x1 + w2x2 + ... + wnxn
   -  So if n_x is large we want W 's to be smaller to not explode the cost.
- So it turns out that we need the variance which equals 1/n_x to be the range of W 's
- So lets say when we initialize W 's like this (better to use with tanh activation):
- np.random.rand(shape) * np.sqrt(1/n[l-1])
- or variation of this (Bengio et al.):
- np.random.rand(shape) * np.sqrt(2/(n[l-1] + n[l]))
- Setting initialization part inside sqrt to 2/n[l-1] for ReLU is better:
- np.random.rand(shape) * np.sqrt(2/n[l-1])
- Number 1 or 2 in the nominator can also be a hyperparameter to tune (but not the first to start with)
- This is one of the best way of partially solution to Vanishing / Exploding gradients (ReLU + Weight Initialization with
variance) which will help gradients not to vanish/explode too quickly
 - The initialization in this video is called "He Initialization / Xavier Initialization" and has been published in 2015 paper.
- 
  
