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

# More Derivative Example:

- f(a) = a^2 ==> d(f(a))/d(a) = 2a
  - a = 5 ==> f(a) = 25
  - a = 5.0001 ==> f(a) = 25.010 approx.
- The differnce is 10 time bigger ,   value after point of a and f(a), consider the 0.1000, and 0.10  
- Derivative is the slope and slope is different in different points in the function thats why the derivative is a
function.

# Computation Graph

The  computation of the neural networks are organised in the term of forward propagation to do the computation and followed by the backword propagation step to compute the gradient descent. The computaion grapgh explains why this is organised in this way

This is the example of from left to right or you can say the forward propagation.

![Computation Graph](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Computation%20grapgh.PNG)


# Computing Derivatives

-Calculus chain rule says: If x -> y -> z (x effect y and y effects z) Then d(z)/d(x) = d(z)/d(y) * d(y)/d(x)
-The video illustrates a big example.
![Computing Derivative](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/computing%20%20derivatives.PNG)
- We compute the derivatives on a graph from right to left and it will be a lot more easier, it is back propagation example.
- dvar means the derivatives of a final output variable with respect to various intermediate quantities.

# logistic Regression Derivatives

-In the video he discussed the derivatives of gradient decent example for one sample with two features x1 and x2.

![logistic Regression derivative](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/logistic%20regression%20derivatives.PNG)


- dz = a-y

# logistic Regression on M examples

- Lets say we have these variables:
X1 Feature
X2 Feature
W1 Weight of the first feature.
W2 Weight of the second feature.
B Logistic Regression parameter.
M Number of training examples
Y(i) Expected output of i

- So we Have

![image](https://user-images.githubusercontent.com/36159918/190208159-6c3d296c-4299-4505-85da-995cdc7a2f39.png)


- Then from right to left we will calculate derivations compared to the result:

d(a) = d(l)/d(a) = -(y/a) + ((1-y)/(1-a))

d(z) = d(l)/d(z) = a - y

d(W1) = X1 * d(z)

d(W2) = X2 * d(z)

d(B) = d(z)

- From the above we can conclude the logistic regression pseudo code:

J = 0; 
dw1 = 0; dw2 =0; db = 0; # Devs.
w1 = 0; w2 = 0; b=0; # Weights

for i = 1 to m
#### Forward pass
     z(i) = W1*x1(i) + W2*x2(i) + b
     a(i) = Sigmoid(z(i))
     J += (Y(i)*log(a(i)) + (1-Y(i))*log(1-a(i)))
#### Backward pass
     dz(i) = a(i) - Y(i)
     dw1 += dz(i) * x1(i)
     dw2 += dz(i) * x2(i)
     db += dz(i)
J /= m
dw1/= m
dw2/= m
db/= m
#### Gradient descent
w1 = w1 - alpa * dw1
w2 = w2 - alpa * dw2
b = b - alpa * db


-It will minimize the error after iteration
- There will be two inner loops for just two input feature map.
- The calculation will be more costly
- To over come this there is  a concept of vectorization

# Vectorization
- Deep learning requeires more and more data. However for loops will take more time to get the result. Thats why we
need vectorization to get rid of some of our for loops.
- NumPy library (dot) function is using vectorization by default.
- The vectorization can be done on CPU or GPU thought the SIMD operation. But its faster on GPU.
- Whenever possible avoid for loops.
- Most of the NumPy library methods are vectorized version.


# Vectorizing Logistic Regression
w' = W transpose
- We will use  only one for loop for gradient descent algorithm , not a sinlge explicit loop
- As an input we have a matrix X and its [Nx, m] and a matrix Y and its [Ny, m] .
- We will then compute at instance [z1,z2...zm] = W' * X + [b,b,...b] . This can be written in python as:

Z = np.dot(W.T,X) + b # Vectorization, then broadcasting, Z shape is (1, m)
A = 1 / 1 + np.exp(-Z) # Vectorization, A shape is (1, m)

- Vectorizing Logistic Regression's Gradient Output:
dz = A - Y # Vectorization, dz shape is (1, m)

dw = np.dot(X, dz.T) / m # Vectorization, dw shape is (Nx, 1)

db = dz.sum() / m # Vectorization, dz shape is (1, 1)

# Notes on Python and NumPy

- In NumPy, obj.sum(axis = 0) sums the columns while obj.sum(axis = 1) sums the rows.
- In NumPy, obj.reshape(1,4) changes the shape of the matrix by broadcasting the values.
- Reshape is cheap in calculations so put it everywhere you're not sure about the calculations.
- Broadcasting works when you do a matrix operation with matrices that doesn't match for the operation, in this case
- NumPy automatically makes the shapes ready for the operation by broadcasting the values.
- In general principle of broadcasting. If you have an (m,n) matrix and you add(+) or subtract(-) or multiply(*) or divide(/)
   with a (1,n) matrix, then this will copy it m times into an (m,n) matrix. The same with if you use those operations with a
   (m , 1) matrix, then this will copy it n times into (m, n) matrix. And then apply the addition, subtraction, and     
    multiplication of division element wise.
- Some tricks to eliminate all the strange bugs in the code:

          - If you didn't specify the shape of a vector, it will take a shape of (m,) and the transpose operation won't work. You have to reshape it to (m, 1)
          - Try to not use the rank one matrix in ANN
          - Don't hesitate to use assert(a.shape == (5,1)) to check if your matrix shape is the required one.
          - If you've found a rank one matrix try to run reshape on it.


# Refrence

- Some of the notes is borrowed from  [Mahmoud Badry](https://github.com/mbadry1/DeepLearning.ai-Summary)
