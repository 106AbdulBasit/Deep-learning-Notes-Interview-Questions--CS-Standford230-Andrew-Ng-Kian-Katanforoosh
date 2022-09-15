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


** Same  process repeats f

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
    
    
