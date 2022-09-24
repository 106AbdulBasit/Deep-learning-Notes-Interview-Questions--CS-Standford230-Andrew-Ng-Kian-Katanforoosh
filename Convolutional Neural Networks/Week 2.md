# Deep convolutional models: case studies

# Why look at case studies?

- We learned about Conv layer, pooling layer, and fully connected layers. It turns out that computer vision researchers
  spent the past few years on how to put these layers together.
- To get some intuitions you have to see the examples that has been made.
- Some neural networks architecture that works well in some tasks can also work well in other tasks.
- Here are some classical CNN networks:
  - LeNet-5
  - AlexNet
  - VGG
- The best CNN architecture that won the last ImageNet competition is called ResNet and it has 152 layers!
- There are also an architecture called Inception that was made by Google that are very useful to learn and apply to your
   tasks.
- Reading and trying the mentioned models can boost you and give you a lot of ideas to solve your task.

# Classic networks
- In this section we will talk about classic networks which are LeNet-5, AlexNet, and VGG.
- **Le Net-5**
    - The goal for this model was to identify handwritten digits in a 32x32x1 gray image. Here are the drawing of it:
    - ![image](https://user-images.githubusercontent.com/36159918/191999874-56ed8714-04f3-4d1d-8d34-dff49a05bf28.png)
    - This model was published in 1998. The last layer wasn't using softmax back then.
    - It has 60k parameters.
    - The dimensions of the image decreases as the number of channels increases.
    - Conv ==> Pool ==> Conv ==> Pool ==> FC ==> FC ==> softmax this type of arrangement is quite common.
    - The activation function used in the paper was Sigmoid and Tanh. Modern implementation uses RELU in most of the
      cases.
- **AlexNet**
  - Named after Alex Krizhevsky who was the first author of this paper. The other authors includes Geoffrey Hinton
  - The goal for the model was the ImageNet challenge which classifies images into 1000 classes. Here are the drawing
    of the model:
  - ![image](https://user-images.githubusercontent.com/36159918/192000263-ce13e8a9-2913-4e90-a71c-0bfae90c9eb1.png)
  - Summary:
    - Conv => Max-pool => Conv => Max-pool => Conv => Conv => Conv => Max-pool ==> Flatten ==> FC ==> FC
      ==> Softmax
  - Similar to LeNet-5 but bigger.
  - Has 60 Million parameter compared to 60k parameter of LeNet-5.
  - It used the RELU activation function.
  - The original paper contains Multiple GPUs and Local Response normalization (RN).
      - Multiple GPUs were used because the GPUs were not so fast back then.
      - Researchers proved that Local Response normalization doesn't help much so for now don't bother yourself for
        understanding or implementing it.
  - This paper convinced the computer vision researchers that deep learning is so important.
- **VGG-16**
  - A modification for AlexNet.
  - Instead of having a lot of hyperparameters lets have some simpler network.
  - Focus on having only these blocks:
     - CONV = 3 X 3 filter, s = 1, same
     - MAX-POOL = 2 X 2 , s = 2
   - Here are the architecture:
   - ![image](https://user-images.githubusercontent.com/36159918/192000866-e356392c-312a-438b-990f-c712bd5692fe.png)

- This network is large even by modern standards. It has around 138 million parameters
  - Most of the parameters are in the fully connected layers.
- It has a total memory of 96MB per image for only forward propagation!
  - Most memory are in the earlier layers.
- Number of filters increases from 64 to 128 to 256 to 512. 512 was made twice.
- Pooling was the only one who is responsible for shrinking the dimensions.
- There are another version called VGG-19 which is a bigger version. But most people uses the VGG-16 instead of the
- VGG-19 because it does the same.
- VGG paper is attractive it tries to make some rules regarding using CNNs.
   
   
   
 # Residual Network
 - Residual Network Are a NN that consists of some Residual blocks.
 - ![image](https://user-images.githubusercontent.com/36159918/192003154-9fff7b87-8f67-4aca-85ab-998aa4d465f4.png)



 - These arrows are the skip connection which are added before the reula ctivation function but after linear function.
 - These networks can go deeper without hurting the performance. In the normal NN - Plain networks - the theory tell
  us that if we go deeper we will get a better solution to our problem, but because of the vanishing and exploding
  gradients problems the performance of the network suffers as it goes deeper. Thanks to Residual Network we can
  go deeper as we want now.
 - ![image](https://user-images.githubusercontent.com/36159918/192003292-51af019a-70c9-48f6-afd4-81ac3424dda6.png)
 - On the left is the normal NN and on the right are the ResNet. As you can see the performance of ResNet increases
    as the network goes deeper
 - In some cases going deeper won't effect the performance and that depends on the problem on your hand.
 - Some people are trying to train 1000 layer now which isn't used in practice.
 
 
 # Why ResNets work
 - Lets see some example that illustrates why resNet work.
  - We have a big NN as the following:
    - X --> Big NN --> a[l]
  - Lets add two layers to this network as a residual block:
    - X --> Big NN --> a[l] --> Layer1 --> Layer2 --> a[l+2]
    -  And a [l] has a direct connection to a[l+2]
  - Suppose we are using RELU activations.
  - Then:
    - a[l+2] = g( z[l+2] + a[l] )
      = g( W[l+2] a[l+1] + b[l+2] + a[l] )
    - Then if we are using L2 regularization for example, W[l+2] will be zero. Lets say that b[l+2] will be zero too.
    - Then a[l+2] = g( a[l] ) = a[l] with no negative values.
    - This show that identity function is easy for a residual block to learn. And that why it can train deeper NNs.
    - Also that the two layers we added doesn't hurt the performance of big NN we made.
    - Hint: dimensions of z[l+2] and a[l] have to be the same in resNets. In case they have different dimensions what we
      put a matrix parameters (Which can be learned or fixed)
    - a[l+2] = g( z[l+2] + ws * a[l] ) # The added Ws should make the dimensions equal
      ws also can be a zero padding.
  - Using a skip-connection helps the gradient to backpropagate and thus helps you to train deeper networks
  - Lets take a look at ResNet on images
    - Here are the architecture of ResNet-34:
    - ![image](https://user-images.githubusercontent.com/36159918/192006652-33861309-ac7a-4694-b502-c8374432b016.png)
    - All the 3x3 Conv are same Convs.
    - Keep it simple in design of the network.
    - spatial size /2 => # filters x2
    - No FC layers, No dropout is used.
    - Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are
    - same or different. You are going to implement both of them.
    - The dotted lines is the case when the dimensions are different. To solve then they down-sample the input by 2 and
    - then pad zeros to match the two dimensions. There's another trick which is called bottleneck which we will explore
      later.
    - Useful concept (Spectrum of Depth):
    - ![image](https://user-images.githubusercontent.com/36159918/192007003-6dcdb334-dec1-4c33-b757-b16e806fd6ba.png)

  - Residual blocks types:
    - Identity block:
      ![image](https://user-images.githubusercontent.com/36159918/192007161-5833ebec-8d4e-4800-bad8-7fac7f54467b.png)
    - Hint the conv is followed by a batch norm BN before RELU . Dimensions here are same.
    - This skip is over 2 layers. The skip connection can jump n connections where n>2
    - This drawing represents Keras layers.
  - The convolutional block:
    - ![image](https://user-images.githubusercontent.com/36159918/192007311-bd81ef4c-0047-4947-a00d-ecfef1e8ff84.png)
    - The conv can be bottleneck 1 x 1 conv


# Network in Network and 1 X 1 convolutions
- A 1 x 1 convolution - We also call it Network in Network- is so useful in many CNN models.
- What does a 1 X 1 convolution do? Isn't it just multiplying by a number?
  - Lets first consider an example:
    - Input: 6x6x1
    - Conv: 1x1x1 one filter. # The 1 x 1 Conv
    - Output: 6x6x1
  - Another example:
    - Input: 6x6x32
    - Conv: 1x1x32 5 filters. # The 1 x 1 Conv
    - Output: 6x6x5
  - The Network in Network is proposed in [Lin et al., 2013. Network in network]
  - It has been used in a lot of modern CNN implementations like ResNet and Inception models.
  - A 1 x 1 convolution is useful when:
    - We want to shrink the number of channels. We also call this feature transformation.
      - In the second discussed example above we have shrinked the input from 32 to 5 channels.
  - We will later see that by shrinking it we can save a lot of computations
  - If we have specified the number of 1 x 1 Conv filters to be the same as the input number of channels then the
  - output will contain the same number of channels. Then the 1 x 1 Conv will act like a non linearity and will learn non
    linearity operator
  - Replace fully connected layers with 1 x 1 convolutions as Yann LeCun believes they are the same.
    - In Convolutional Nets, there is no such thing as "fully-connected layers". There are only convolution layers with
      1x1 convolution kernels and a full connection table. Yann LeCun
 
 # Inception network motivation
 
 - When you design a CNN you have to decide all the layers yourself. Will you pick a 3 x 3 Conv or 5 x 5 Conv or maybe a
   max pooling layer. You have so many choices.
 -  What inception tells us is, Why not use all of them at once?
 - Inception module, naive version:
     ![image](https://user-images.githubusercontent.com/36159918/192020156-8940402f-191b-4adf-8add-6f4fe1311390.png)
 - Hint that max-pool are same here.
 - Input to the inception module are 28 x 28 x 192 and the output are 28 x 28 x 256
 - We have done all the Convs and pools we might want and will let the NN learn and decide which it want to use
   most.
 - The problem of computational cost in Inception model:
    - If we have just focused on a 5 x 5 Conv that we have done in the last example.
    - There are 32 same filters of 5 x 5, and the input are 28 x 28 x 192.
    - Output should be 28 x 28 x 32
    - The total number of multiplications needed here are:
      - Number of outputs * Filter size * Filter size * Input dimensions
      - Which equals: 28 * 28 * 32 * 5 * 5 * 192 = 120 Mil
      - 120 Mil multiply operation still a problem in the modern day computers.
      - Using a 1 x 1 convolution we can reduce 120 mil to just 12 mil. Lets see how.
    - Using 1 X 1 convolution to reduce computational cost:
      - The new architecture are:
         - X0 shape is (28, 28, 192)
         - We then apply 16 (1 x 1 Convolution)
         - That produces X1 of shape (28, 28, 16)
         - Hint, we have reduced the dimensions here.
         - Then apply 32 (5 x 5 Convolution)
         - That produces X2 of shape (28, 28, 32)
      - Now lets calculate the number of multiplications:
        - For the first Conv: 28 * 28 * 16 * 1 * 1 * 192 = 2.5 Mil
        - For the second Conv: 28 * 28 * 32 * 5 * 5 * 16 = 10 Mil
        - So the total number are 12.5 Mil approx. which is so good compared to 120 Mil
      - A 1 x 1 Conv here is called Bottleneck BN .
      - It turns out that the 1 x 1 Conv won't hurt the performance.
      - Inception module, dimensions reduction version:
        ![image](https://user-images.githubusercontent.com/36159918/192023433-c556a432-db5f-439d-8193-06269740167d.png)
      - Example of inception model in Keras:
         ![image](https://user-images.githubusercontent.com/36159918/192023605-db8aa7b7-ca80-4bf0-98e6-532060f61121.png)


# Inception network (GoogleNet)
- The inception network consist of concatenated blocks of the Inception module.
- The name inception was taken from a meme image which was taken from Inception movie
- Here are the full model:
  ![image](https://user-images.githubusercontent.com/36159918/192089687-e0d15816-2050-411b-b1f5-18d1ee1681e7.png)
- Some times a Max-Pool block is used before the inception module to reduce the dimensions of the inputs.
- There are a 3 Sofmax branches at different positions to push the network toward its goal. and helps to ensure that the
  intermediate features are good enough to the network to learn and it turns out that softmax0 and sofmax1 gives
  regularization effect.
- Since the development of the Inception module, the authors and the others have built another versions of this network.
  Like inception v2, v3, and v4. Also there is a network that has used the inception module and the ResNet together.


# Mobile Net

 **Motivation for Mobile Net**
-	Low computational cost at deployment
-	Useful for mobile and embedded vision application
-	Key idea: Normal vs Depthwise- separable convolutions
- Normal Convolutions
  - Nx * nx * nc(6 * 6 * 3)  >> filter( 3*3*)  f * f * nc = (4 *  4 * 5)

  - Computational Cost( 2160)  = (3  * 3 * 3 ) filter parameters    *  (4 * 4) filter position    * (5) number of filters
- Depth wise Seprable Convolution
   - Each filter seprable will be multiply to one of the input channel
   - nx * nx * nc convolve 3*3(filter) >>  nout * nout * nc
   - Computational Cost( 432)  = (3  * 3  ) filter parameters    *  (4 * 4) filter position    * (3) number of filters
   - There's one more step, which is, we need to take this 4 by 4 by 3 intermediate value and carry out one more step. 
- Point Wise Convolution
  - (n out  * n out * nc ) Convolve (1*1*nc(3)) >> when nc' is   five filters we get >> (4 * 4 * 5)
  -  Computational Cost( 240)  = (1 * 1 * 3 ) filter parameters    *  (4 * 4) filter position    * (5) number of filters 
- Cost Summary
  - Cost of Normal convolution = 2160
  - Cost of Depth wise seperarbel convoultion
      - Depth wise = 432
      - poit wise = 240
      - Total 432 + 240 = 672
  - 672 / 2160 = 0.31
  -  the depthwise separable convolution was about 31 percent as computationally expensive as the normal convolution, roughly 3x savings.
  - The Genral Ratio rule is
    - 1/nc' + 1/f^2
    - In our example >>  1/5 + 1/9 = 0.31
    - Normally the input channel are much bigger
    - For instance if you have 512 then
    - 1/512 + 1/9
 - the depthwise separable convolution rougly 10 times cheaper in computational costs

# Mobile Net Architecture
- In mobile net architecture it has 13 13 blocks of mobile net followed by pooling Fully connected and softmax
- In mobile net v2 we have  Residual blocks
- ![Mobilenet](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/MobileNet%20v2.PNG)

- The first step is exapnasion  you go from n*n*3 >> n *n * 18
- Normally the expansion is done with factor of 6
- The secod step is Depth wise convolution
-  This last step is also called a projection step because you're projecting down from n by n by 18 down to n by n by 3.
-  The expansion allow the neural network to learn more richer function
-   You want to deploy it to the mobile devices that's why the projection use to back down to a smaller set of value
-  allow your neural network to learn richer and more complex functions, while also keeping the amounts of memory that is the size of the activations you need to      pass from layer to layer
-  Mobile net comparsions
  - ![Comparsion](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/MObilenetcomparsion.PNG)


# Efficent Net
- there a way to tune MobileNet, or some other architecture, to your specific device
- We can Scale the neural network in to depth or resolution or in dimensions
- Can we scale up neural netrwok automatically
-  EfficientNet, gives you a way to do so
-  We have a base line archetecture and we will scale in to to three dimensions according to our need
-  ![Efiecent](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Efficent%20Net.PNG)

#  Using Open-Source Implementation
- We have learned a lot of NNs and ConvNets architectures.
- It turns out that a lot of these NN are difficult to replicated. because there are some details that may not presented on
  its papers. There are some other reasons like:
  - Learning decay.
  - Parameter tuning.
- lot of deep learning researchers are opening sourcing their code into Internet on sites like Github.
- If you see a research paper and you want to build over it, the first thing you should do is to look for an open source
   implementation for this paper.
- Some advantage of doing this is that you might download the network implementation along with its
  parameters/weights. The author might have used multiple GPUs and spent some weeks to reach this result and its right
  in front of you after you download it.
  
# Transfer Learning
- If you are using a specific NN architecture that has been trained before, you can use this pretrained parameters/weights
  instead of random initialization to solve your problem.
- It can help you boost the performance of the NN.
  The pretrained models might have trained on a large datasets like ImageNet, Ms COCO, or pascal and took a lot of time
  to learn those parameters/weights with optimized hyperparameters. This can save you a lot of time.
- Lets see an example:
  - Lets say you have a cat classification problem which contains 3 classes Tigger, Misty and neither.
  - You don't have much a lot of data to train a NN on these images.
  - Andrew recommends to go online and download a good NN with its weights, remove the softmax activation layer
    and put your own one and make the network learn only the new layer while other layer weights are fixed/frozen.
  - Frameworks have options to make the parameters frozen in some layers using trainable = 0 or freeze = 0
  - One of the tricks that can speed up your training, is to run the pretrained NN without final softmax layer and get an
    intermediate representation of your images and save them to disk. And then use these representation to a shallow
    NN network. This can save you the time needed to run an image through all the layers.
      - Its like converting your images into vectors.
  - Another example
    - What if in the last example you have a lot of pictures for your cats.
    - One thing you can do is to freeze few layers from the beginning of the pretrained network and learn the other
      weights in the network.
    - Some other idea is to throw away the layers that aren't frozen and put your own layers there.
   - Another example:
      - If you have enough data, you can fine tune all the layers in your pretrained network but don't random initialize the
        parameters, leave the learned parameters as it is and learn from there.
     
   # Data Augmentation
   - If data is increased, your deep NN will perform better. Data augmentation is one of the techniques that deep learning
      uses to increase the performance of deep NN.
   - The majority of computer vision applications needs more data right now.
   - Some data augmentation methods that are used for computer vision tasks includes:
      - Mirroring.
      - Random cropping.
        - The issue with this technique is that you might take a wrong crop.
        - The solution is to make your crops big enough. 
      - Rotation.
      - Shearing.
      - Local warping.
      - Color shifting.
          - For example, we add to R, G, and B some distortions that will make the image identified as the same for the
            human but is different for the computer.
          - In practice the added value are pulled from some probability distribution and these shifts are some small.
          - Makes your algorithm more robust in changing colors in images.
          - There are an algorithm which is called PCA color augmentation that decides the shifts needed automatically.   
   - Implementing distortions during training:
      - You can use a different CPU thread to make you a distorted mini batches while you are training your NN.
   - Data Augmentation has also some hyperparameters. A good place to start is to find an open source data augmentation
    implementation and then use it or fine tune these hyperparameters.
  
  
  # State of Computer Vision
  - For a specific problem we may have a little data for it or a lots of data.
  - Speech recognition problems for example has a big amount of data, while image recognition has a medium amount of
    data and the object detection has a small amount of data nowadays.
  - If your problem has a large amount of data, researchers are tend to use:
    - Simpler algorithms.
    - Less hand engineering.
  - If you don't have that much data people tend to try more hand engineering for the problem "Hacks". Like choosing a
      more complex NN architecture.
   - Because we haven't got that much data in a lot of computer vision problems, it relies a lot on hand engineering. 
   - We will see in the next chapter that because the object detection has less data, a more complex NN architectures will be
      presented.
   - Tips for doing well on benchmarks/winning competitions:
      - Ensembling.
        - Train several networks independently and average their outputs. Merging down some classifiers.
        - After you decide the best architecture for your problem, initialize some of that randomly and train them
            independently.
        - This can give you a push by 2%
        - But this will slow down your production by the number of the ensembles. Also it takes more memory as it saves
            all the models in the memory.
        - People use this in competitions but few uses this in a real production.
       - Multi-crop at test time.
          - Run classifier on multiple versions of test versions and average results.
          - There is a technique called 10 crops that uses this.
          - This can give you a better result in the production.
   - Use open source code
    - Use architectures of networks published in the literature.
    - Use open source implementations if possible.
    - Use pretrained models and fine-tune on your dataset.
     
