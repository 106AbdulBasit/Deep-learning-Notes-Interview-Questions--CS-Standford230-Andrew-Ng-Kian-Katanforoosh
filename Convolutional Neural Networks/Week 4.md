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
    - 
