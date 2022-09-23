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


