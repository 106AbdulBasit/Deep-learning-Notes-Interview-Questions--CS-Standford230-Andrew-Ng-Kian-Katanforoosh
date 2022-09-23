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


