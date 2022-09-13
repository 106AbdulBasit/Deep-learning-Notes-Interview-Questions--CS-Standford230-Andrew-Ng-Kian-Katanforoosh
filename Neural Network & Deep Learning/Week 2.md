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

