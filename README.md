# Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh
The aim of the repository is to make the resource in which all the course notes will be covered and with possible interviwe questions as well.



# Lecture 1

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



 
 

 
