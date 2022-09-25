# Face Recognition

# What is face recognition?
- Face recognition system identifies a person's face. It can work on both images or videos.
- Liveness detection within a video face recognition system prevents the network from identifying a face in an image. It
   can be learned by supervised deep learning using a dataset for live human and in-live human and sequence learning.
- Face verification vs. face recognition:
    - Verification:
       - Input: image, name/ID. (1 : 1)
       - Output: whether the input image is that of the claimed person.
       - "is this the claimed person?
     - Recognition:
        - Has a database of K persons
        - Get an input image
        - Output ID if the image is any of the K persons (or not recognized)
        - "who is this person?"
- We can use a face verification system to make a face recognition system. The accuracy of the verification system has to
be high (around 99.9% or more) to be use accurately within a recognition system because the recognition system
accuracy will be less than the verification system given K persons.

# One Shot Learning

- One of the face recognition challenges is to solve one shot learning problem.
- One Shot Learning: A recognition system is able to recognize a person, learning from one image.
- Historically deep learning doesn't work well with a small number of data.
- Instead to make this work, we will learn a similarity function:
  - d( img1, img2 ) = degree of difference between images.
  - We want d result to be low in case of the same faces.
  - We use tau T as a threshold for d:
    - If d( img1, img2 ) <= T Then the faces are the same.
  - Similarity function helps us solving the one shot learning. Also its robust to new inputs.
  
  
  # Siamese Network
  
  - We will implement the similarity function using a type of NNs called Siamease Network in which we can pass multiple
    inputs to the two or more networks with the same architecture and parameters.
  - Siamese network architecture are as the following:
    - ![image](https://user-images.githubusercontent.com/36159918/192159775-f45b5797-a45b-440a-ac3a-57a6cb31ca7e.png)
  - We make 2 identical conv nets which encodes an input image into a vector. In the above image the vector shape is
   (128, )
  - The loss function will be d(x1, x2) = || f(x1) - f(x2) ||^2
    If X1 , X2 are the same person, we want d to be low. If they are different persons, we want d to be high.
  - [Taigman et. al., 2014. DeepFace closing the gap to human level performance]
  
  # Triplet Loss
  
  - Triplet Loss is one of the loss functions we can use to solve the similarity distance in a Siamese network.
  - Our learning objective in the triplet loss function is to get the distance between an Anchor image and a positive or a
    negative image.
      -  Positive means same person, while negative means different person.
  - The triplet name came from that we are comparing an anchor A with a positive P and a negative N image
  - Formally we want:
      - ||f(A) - f(P)||^2 <= ||f(A) - f(N)||^2
      - Then
      - ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 <= 0
      - To make sure the NN won't get an output of zeros easily:
      - ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 <= -alpha
      - Alpha is a small number. Sometimes its called the margin.
      - Then
         ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha <= 0
   - Final Loss function:
      - Given 3 images (A, P, N)
      - L(A, P, N) = max (||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha , 0)
      - J = Sum(L(A[i], P[i], N[i]) , i) for all triplets of images.
   - You need multiple images of the same person in your dataset. Then get some triplets out of your dataset. Dataset
     should be big enough.
   - Choosing the triplets A, P, N:
      - During training if A, P, N are chosen randomly (Subjet to A and P are the same and A and N aren't the same) then
      - one of the problems this constrain is easily satisfied
      -  d(A, P) + alpha <= d (A, N)
      - So the NN wont learn much
    - What we want to do is choose triplets that are hard to train on.
        - So for all the triplets we want this to be satisfied:
        - d(A, P) + alpha <= d (A, N)
        - This can be achieved by for example same poses!
        - Find more at the paper
     - Details are in this paper [Schroff et al.,2015, FaceNet: A unified embedding for face recognition and clustering]
     - Commercial recognition systems are trained on a large datasets like 10/100 million images.
     - There are a lot of pretrained models and parameters online for face recognition.
    
    
  # Face Verification and Binary Classification
  
  - Triplet loss is one way to learn the parameters of a conv net for face recognition there's another way to learn these
    parameters as a straight binary classification problem.
  -  Learning the similarity function another way:
      - ![image](https://user-images.githubusercontent.com/36159918/192160049-8f14be0e-fedc-4b8a-b59a-7b6583cc1a19.png)

  - The final layer is a sigmoid layer.
  - Y' = wi * Sigmoid ( f(x(i)) - f(x(j)) ) + b where the subtraction is the Manhattan distance between f(x(i)) and
      f(x(j))
   - Some other similarities can be Euclidean and Ki square similarity.
   - The NN here is Siamese means the top and bottom convs has the same parameters.
    - The paper for this work: [Taigman et. al., 2014. DeepFace closing the gap to human level performance]
    - A good performance/deployment trick:
      - Pre-compute all the images that you are using as a comparison to the vector f(x(j))
      - When a new image that needs to be compared, get its vector f(x(i)) then put it with all the pre computed vectors
          and pass it to the sigmoid function.
     - This version works quite as well as the triplet loss function
     - Available implementations for face recognition using deep learning includes:
        -  Openface
        - FaceNet
        - DeepFace
