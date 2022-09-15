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






    
    
    
    
    
    
