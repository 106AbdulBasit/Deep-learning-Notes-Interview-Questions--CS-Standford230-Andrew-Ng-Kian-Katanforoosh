# Week 1

##  Question -What is Neural Network

To understand the neural network lets have a look at the simplest form of the neural network


![Example2](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Single%20Neuron.PNG)

The input to the neural network is the size of the house which is denoted by the x.  In the middle, we have the circle which is the single neuron and input goes into this neuron and it output the price which is denoted as Y. The neuron implements the function which is drawn on the left. This function takes the max of zero and then outputs the estimated price. 

So input goes into neuron , the neuron computes the linear function and output the prediction(estimated price).  
 In this example the linear function is Rectified Linear Unit.

 Now lets have a look at the relatively a big neural network

 ![Simple Neural Networl](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Simple%20Neural%20Network.PNG)


As you can see instead of one neuron and one input we have multiple neurons and inputs.  These for input are represented as the input features. In this example we have four input features, X1 represents the size, X2 represents the number of bedrooms, X3 represents the zip code and X4 represents the wealth.  These neurons call the hidden units. If you can focus on the right top of the image, you can say that the first neuron may compute the family size by taking the first two inputs, similarly, the second neuron may compute the wealth of the area. Assuming that the first node represents the family size and family size depends on the first two inputs we going to let the neural network decide whatever nodes this is known to be.

 So we can say that X is the Input layer and they are densely connected to the neural layer. We are saying the densely connected layer because every input is connected to every neuron in the middle layer.


 # Supervised Learning

 In supervised learning you have input x and you  have a function mapping that on output Y.  Below there are some examples of the Supervised Learning

 ![Supervised Learning](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Supervised%20Learning.PNG)


The different types of neural architecture used for different task

 The first two tasks use the standard neural network.

 For Image recognition, the CNN is more useful

 For  Audio to text script and for machine translation we use RNN, it turns out RNN performs well for sequence data.

 For Complex task like autonomous driving ,you might end up with hybrid Neural network architecture

 Below there are images of the architecture.


 ![NN Types](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Neural%20Network%20Examples.PNG)

 There are two types of Data:
 - Structered Data
 - Unstructurede Data

![Structure and Unstructure Data](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Structered%20%26%20Unstruct%20data.PNG)

### Structure Data
For example you have data base of houses which has rows and columns , this is a structured Data

### Unstructure Data

The audio files are examples of the unstructured data, Images are also examples

Historically the computer performs well on structured data and not able to perform well on unstructured data. Now with the help of neural networks now computers are able to perform well on unstructured data.

 218  
Neural Network & Deep Learning/Week 2.md
@@ -1,217 +1 @@
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
Notesss
 222  
Neural Network & Deep Learning/Week 3.md
@@ -1,222 +0,0 @@
# Neural Network Overview

- In logistic regression the architeture looks like this :


X1 \
X2 ==> z = XW + B ==> a = Sigmoid(z) ==> l(a,Y)
X3 /

x are the input features to the z which is function followed by the sigmoid activation function and the calculate the loss

- In Neural Network :

X1 \
X2 => z1 = XW1 + B1 => a1 = Sigmoid(z1) => z2 = a1W2 + B2 => a2 = Sigmoid(z2) => l(a2,Y)
X3 /

- Neural Netwrok is the stack of logistic regression objects.


# Neural Netwrok Representation:

![NNrpre](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Week%203/Neural%20Network%20Representation.PNG)

- We will define the neural networks that has one hidden layer.
- NN contains of input layers, hidden layers, output layers.
- Hidden layer means we cant see that layers in the training set.
- a0 = x (the input layer)
- a2 will represent the output layer.
- This has two layers we dont count the input layer

 # Neural Network Representation computation:

 ![NNC](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Week%203/Neural%20Network%20Representation%20computation.PNG)

In logistic regression the compuation is in two parst
- z =  w't(x)+b
- a = sigmoid of z >> this is activation function 
- This is process of each neouron


** Same  process repeats for other neourons**

- Some  description of image:
    - Number of neourons = 4
    - Number of inputs nx = 3
- Shapes of the variables:
    - W1 is the matrix of the first hidden layer, it has a shape of (noOfHiddenNeurons,nx)
    - b1 is the matrix of the first hidden layer, it has a shape of (noOfHiddenNeurons,1)
    - z1 is the result of the equation z1 = W1*X + b , it has a shape of (noOfHiddenNeurons,1)
    - a1 is the result of the equation a1 = sigmoid(z1) , it has a shape of (noOfHiddenNeurons,1)
    - W2 is the matrix of the second hidden layer, it has a shape of (1,noOfHiddenNeurons)
    - b2 is the matrix of the second hidden layer, it has a shape of (1,1)
    - z2 is the result of the equation z2 = W2*a1 + b , it has a shape of (1,1)
    - a2 is the result of the equation a2 = sigmoid(z2) , it has a shape of (1,1)


    Vectorization acroos M  Examples

    - pesudo code of forward propagation of two layers



'''
for i = 1 to m
      z[1, i] = W1*x[i] + b1 # shape of z[1, i] is (noOfHiddenNeurons,1)
      a[1, i] = sigmoid(z[1, i]) # shape of a[1, i] is (noOfHiddenNeurons,1)
      z[2, i] = W2*a[1, i] + b2 # shape of z[2, i] is (1,1)
      a[2, i] = sigmoid(z[2, i]) # shape of a[2, i] is (1,1)

'''




- Lets say we have X on shape (Nx,m) . So the new pseudo code:

    '''
    Z1 = W1X + b1 # shape of Z1 (noOfHiddenNeurons,m)
    A1 = sigmoid(Z1) # shape of A1 (noOfHiddenNeurons,m)
    Z2 = W2A1 + b2 # shape of Z2 is (1,m)
    A2 = sigmoid(Z2) # shape of A2 is (1,m)
    '''

   - In the last example we can call X = A0 . So the previous step can be rewritten as:

'''
Z1 = W1A0 + b1 # shape of Z1 (noOfHiddenNeurons,m)
   A1 = sigmoid(Z1) # shape of A1 (noOfHiddenNeurons,m)
   Z2 = W2A1 + b2 # shape of Z2 is (1,m)
   A2 = sigmoid(Z2) # shape of A2 is (1,m)
'''


# Activation Functions

![af](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Week%203/Activation%20functions.PNG)

- Till now we are using sigmoid activation function, but in some cases other functions can be a lot better.
- Sigmoid can lead us to gradient decent problem where the updates are so low.
- Tanh activation function range is [-1,1] (Shifted version of sigmoid function)
- It turns out that the tanh activation usually works better than sigmoid activation function for hidden units because the
   mean of its output is closer to zero, and so it centers the data better for the next layer.
- Sigmoid or Tanh function disadvantage is that if the input is too small or too high, the slope will be near zero which will
   cause us the gradient decent problem
- One of the popular activation functions that solved the slow gradient decent is the RELU function. RELU = max(0,z) # so
   if z is negative the slope is 0 and if z is positive the slope remains linear.
- So here is some basic rule for choosing activation functions, if your classification is between 0 and 1, use the output
  activation as sigmoid and in other layers uses tan h activation function.
- if you dont know that which activation function to choose , go with relu activation function
- Leaky RELU activation function different of RELU is that if the input is negative the slope will be so small. It works as
RELU but most people uses RELU. Leaky_RELU = max(0.01z,z) #the 0.01 can be a parameter for your algorithm.

# Why do we need non linear activation function

- we removed the activation function from our algorithm that can be called linear activation function.
- Linear activation function will output linear activations
- There will be no complex calculation the output might be same.
- Whatever hidden layers you add, the activation will be always linear like logistic regression (So its useless in a lot of
complex problems)
- You might use linear activation function in one place - in the output layer if the output is real numbers (regression
problem). But even in this case if the output value is non-negative you could use RELU instead.



# Derivatives of the activation Functions

**Sigmoid**

'''
g(z) = 1 / (1 + np.exp(-z))
g'(z) = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
g'(z) = g(z) * (1 - g(z))
'''

**Tanh activation function**
'''
g(z) = (e^z - e^-z) / (e^z + e^-z)
g'(z) = 1 - np.tanh(z)^2 = 1 - g(z)^2
'''

**Derivative of Relu activation Function**
'''
g(z) = np.maximum(0,z)
g'(z) = { 0 if z < 0
1 if z >= 0 }
'''
**leakyRelu activation Function**
'''
g(z) = np.maximum(0.01 * z, z)
g'(z) = { 0.01 if z < 0
1 if z >= 0 }
'''



# Gradeint Descent for the neural Network:

- Gradient descent algorithm:
  - NN parameters
    - n[0] = Nx
    - n[1] = NoOfHiddenNeurons
    - n[2] = NoOfOutputNeurons = 1
    - W1 shape is (n[1],n[0])
    - b1 shape is (n[1],1)
    - W2 shape is (n[2],n[1])
    - b2 shape is (n[2],1)
  - Cost function I = I(W1, b1, W2, b2) = (1/m) * Sum(L(Y,A2))
  - Gradeint Descent Algorithm
  '''
  Repeat:
Compute predictions (y'[i], i = 0,...m)
Get derivatives: dW1, db1, dW2, db2
Update: W1 = W1 - LearningRate * dW1
b1 = b1 - LearningRate * db1
W2 = W2 - LearningRate * dW2
b2 = b2 - LearningRate * db2
'''

- Forward propagation:
'''
Z1 = W1A0 + b1 # A0 is X
A1 = g1(Z1)
Z2 = W2A1 + b2
A2 = Sigmoid(Z2) # Sigmoid because the output is between 0 and 1
'''
- Backpropagation (derivations):
'''
dZ2 = A2 - Y # derivative of cost function we used * derivative of the sigmoid function
dW2 = (dZ2 * A1.T) / m
db2 = Sum(dZ2) / m
dZ1 = (W2.T * dZ2) * g'1(Z1) # element wise product (*)
dW1 = (dZ1 * A0.T) / m # A0 = X
db1 = Sum(dZ1) / m
# Hint there are transposes with multiplication because to keep dimensions correc
'''

# Random initialization

- In logistic regression it wasn't important to initialize the weights randomly, while in NN we have to initialize them
randomly.
- If we initialize all the weights with zeros in NN it won't work (initializing bias with zero is OK):
   - all hidden units will be completely identical (symmetric) - compute exactly the same function
   - on each gradient descent iteration all the hidden units will always update the same.
- We need small values because in sigmoid (or tanh), for example, if the weight is too large you are more likely to end up
even at the very start of training with very large values of Z. Which causes your tanh or your sigmoid activation function
to be saturated, thus slowing down learning. If you don't have any sigmoid or tanh activation functions throughout your
neural network, this is less of an issue.
- Constant 0.01 is alright for 1 hidden layer networks, but if the NN is deep this number can be changed but it will always
be a small number.












 154  
Neural Network & Deep Learning/Week 4.md
@@ -1,154 +0,0 @@
# Deep Neural Network

- A shallow neural network is one which has two neural layers.
- A Deep neural network is one which has more than two layers and it can be many.
- We will use the notation L to denote the number of layers in a NN.
- n[l] is the number of neurons in a specific layer l .
- n[0] denotes the number of neurons input layer. n[L] denotes the number of neurons in output layer.
- g[l] is the activation function.
- a[l] = g[l](z[l])
- w[l] weights is used for z[l]
- x = a[0] , a[l] = y'
- We will use the notation L to denote the number of layers in a NN.
- So we have:
  - A vector n of shape (1, NoOfLayers+1)
  - A vector g of shape (1, NoOfLayers)
  - A list of different shapes w based on the number of neurons on the previous and the current layer.
  - A list of different shapes b based on the number of neurons on the current layer.



 # Forward Propagation in a Deep Network

 This is the representation of the forward propagation in a deep network

 ![FP](https://github.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/blob/main/Images/week4/forward%20propagation.PNG)

 - Forward propagation general rule for one input:
 '''
 z[l] = W[l]a[l-1] + b[l]
a[l] = g[l](a[l])
 '''

 - Forward propagation genral rule for m trainging examples

 '''
 Z[l] = W[l]A[l-1] + B[l]
  A[l] = g[l](A[l])
 '''

 - There is no way  that you can compute the forward propagation with out any for loop , so it is okay to have one for loop
 - The matix dimensions need to be watch carefully

 # Getting your matrix dimensions right

 In implementing the deep neural network you have to be carefull about the dimensions

 ![Dimension](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/week4/Dimesnions.PNG)

 - The best way to debug your matrices dimensions is by a pencil and paper.
 - Dimension of W is (n[l],n[l-1]) . Can be thought by right to left.
 - Dimension of b is (n[l],1)
 - In Back Propagation dw has the same shape as W , while db is the same shape as b
 - Dimension of Z[l], A[l] , dZ[l] , and dA[l] is (n[l],m)

 # Why deep representations?

 ![Dep Represesntation](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/week4/deep%20representation.PNG)

 - It turns out the early layer detect the edges  and the more you go deeper the layer will detect the more complex feature like face.
 - The starting function of layer take the small size of the image and the deeper layer take the more size of image as compare to early ones
 - Circuit theory and deep learning
    -  So informally, their functions compute with a relatively small but deep neural network and by small I mean the number of hidden units is relatively small. 
    But if you try to compute the same function with a shallow network, 
 - When starting on an application don't start directly by dozens of hidden layers. Try the simplest solutions (e.g. Logistic
Regression), then try the shallow neural network and so on.
    so if there aren't enough hidden layers, then you might require exponentially more hidden units to compute



# Building blocks of deep neural networks

- Building blocks of deep neural networks
- 
![image](https://user-images.githubusercontent.com/36159918/190650935-1d028b1a-8512-48a4-8d64-09b2349e167e.png)

- Deep NN Block

![image](https://user-images.githubusercontent.com/36159918/190654086-2407694f-f5b5-4466-ad58-61dda09415ba.png)


- The blue line shows about the forward propagation
- The red line shows the back word propagation


# Forward and Backward Propagation

- Pseudo code for forward propagation for layer l:
'''
Input A[l-1]
Z[l] = W[l]A[l-1] + b[l]
A[l] = g[l](Z[l])
Output A[l], cache(Z[l])
'''
- Pseudo code for back propagation for layer l:
'''
Input da[l], Caches
dZ[l] = dA[l] * g'[l](Z[l])
dW[l] = (dZ[l]A[l-1].T) / m
db[l] = sum(dZ[l])/m # Dont forget axis=1, keepdims=True
dA[l-1] = w[l].T * dZ[l] # The multiplication here are a dot product.
Output dA[l-1], dW[l], db[l]
'''

- The deraviative of with respect to the loss function

'''
 dA[L] = (-(y/a) + ((1-y)/(1-a)))
'''

# Parameters vs Hyperparameters
- **Parameters**
  - Weights
  - Bias
- ** Hyperparameters**
  - Learning rate.
  - Number of iteration.
  - Number of hidden layers L .
  - Number of hidden units n .
  - Choice of activation functions.

- Change the hyperparamters
- try the different value
- the value which is valid for today may be is not valid after one year
- In the earlier days of DL and ML learning rate was often called a parameter, but it really is (and now everybody call it) a
hyperparameter.


# Total Number of Parameters
- The total number of parameters is the sum of all the weights and biases on the neural network. When calculating manually, different types of layers have different methods. The parameters on the Dense, Conv2d, or maybe LSTM layers are slightly different. The principle is the same, we only need to calculate the unit weight and bias.

Consider this image

![pm](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/week4/Denselayer.png)

As shown in illustration 1, on the input layer we have 4 input units. And in the hidden layer we have a dense layer with 2 units. Lets say at input layer we have X = {x1, x2, x3, x4}, and in the hidden layer we have a1, a2.

a1 = x1.w11 + x2.w12 + x3.w13 + x4.w14 + b1

a2 = x1.w21 + x2.w22 + x3.w23 + x4.w24 + b2

from the equation it is found that the sum of all weights is 8 which consist of all W= {w11, w12, w13, w14, w21, w22, w23, w24}, and the bias that consist of B = {b1, b2}. Then the total weight and bias is 8+2=10 parameter. If we check it using tensorflow we will get the same amount.

![image](https://user-images.githubusercontent.com/36159918/190663416-09fa615c-ecfa-48de-b461-6904728d5ac8.png)

![image](https://user-images.githubusercontent.com/36159918/190663524-5c63f9d9-c535-4fba-b908-9ca2551776bb.png)

# What does this have to do with the brain

![brain](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/week4/brainminin.PNG)

- The analogy that "It is like the brain" has become really an oversimplified explanation.
- No human today know exactly how many neurons on the brain.
- There is a very simplistic analogy between a single logistic unit and a single neuron in the brain.
- NN is a small representation of how brain work. The most near model of human brain is in the computer vision (CNN)
