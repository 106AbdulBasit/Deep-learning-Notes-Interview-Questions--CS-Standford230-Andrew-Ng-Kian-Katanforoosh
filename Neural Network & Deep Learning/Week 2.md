# Binary Classification
In binary classification , the network try to classsifies that either the image is the labeled one or not.

See the following image.

![Binary Classification](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Binary%20Classification.PNG)

In the above image the task of the network is to classify image that wether the image is of cat or not?

But in order to do the computaion we have to change the pixel value into feature of X.

Assuming the image is  the size of 64 x 64 and it has three channels the Red, Green Blue. All the pixel value will be flattened in to vector x  in order to do the computatioon for the neural network

Some of the notations

- M is the number of training vectors
- Nx is the size of the input vector
- Ny is the size of the output vector
- X(1) is the first input vector
- Y(1) is the first output vector
- X = [x(1) x(2).. x(M)]
- Y = (y(1) y(2).. y(M))

# Logistic Regression

![Logistic regression](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/logistic%20regression.PNG)

- In linear  Regression we have a  Simple equation: y = wx + b
- In Classification you have to give the probablities , but by using the simple linear regression it is not possible , the value may be go higher then 1 or may be go in negative.
- If we need to have the probablities between 0 and 1 we will use sigmoid function
- In the image z = w(t)x + b
- if z  is large then sigmoid of z is closer to 1
- if z is large in negative number then sigmoid of z will be zero

# Difference between Linear regression and logistic regression.

# logistic regression cost function.

- The loss function which can be used is squared  root error which is = L(y',y) = 1/2 (y' - y)^2
     - It turns out that this loss function will not find the local optima in the gradient descent and function will not convex.
- The loss function we will use : L(y',y) = - (y*log(y') + (1-y)*log(1-y'))
- To explain the last function lets see:
    - if y = 1 ==> L(y',1) = -log(y') ==> we want y' to be the largest ==> y ' biggest value is 1
    - if y = 0 ==> L(y',0) = -log(1-y') ==> we want 1-y' to be the largest ==> y' to be smaller as possible
      because it can only has 1 value.
-  The cost function will be :  J(w,b) = (1/m) * Sum(L(y'[i],y[i])) 

# What is the differnce between cost function and loss function?
The loss function computes the error for a single training example; the cost function is the average of the loss functions
of the entire training set.

# Gradient Descent.
![GD](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Gradient%20Descent.PNG)

-We want to have the value of w and b which minimise the cost function.
- Our cost function is convex means it will find the optimal minimum value  of the slope.
-  First step is to initialize the W and b, The value of the W and B can be 0,0 or it can be a random value. It can be both negative or positive.
-  The convex function will try to improve the value to reach the global minimum.
-  In Logistic regression people always use 0,0 instead of random.
-  The gradient decent algorithm repeats: w = w - alpha * dw where alpha is the learning rate and dw is the derivative of
   w (Change to w ) The derivative is also the slope of w.
- Looks like greedy algorithms. the derivative give us the direction to improve our parameters.
- The implementation looks like:
    - w = w - alpha * d(J(w,b) / dw) (how much the function slopes in the w direction)
    - b = b - alpha * d(J(w,b) / db) (how much the function slopes in the d direction)

# Derivatives

- We will talk about some of required calculus.
- Slope = Height / Width
- Derivative of a linear line is its slope.
  ex. f(a) = 3a d(f(a))/d(a) = 3
- if a = 2 then f(a) = 6
-if we move a a little bit a = 2.001 then f(a) = 6.003 means that we multiplied the derivative (Slope) to the
moved area and added it to the last result.


