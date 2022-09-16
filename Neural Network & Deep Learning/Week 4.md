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
