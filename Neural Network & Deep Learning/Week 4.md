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
