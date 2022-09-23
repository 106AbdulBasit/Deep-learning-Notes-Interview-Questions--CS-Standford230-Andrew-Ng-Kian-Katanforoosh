# Computer vision

- Computer vision is one of the applications that are rapidly active thanks to deep learning
- Some of the applications of computer vision that are using deep learning includes:
  - Self driving cars.
  - Face recognition.
- Deep learning is also enabling new types of art to be created.
- Rapid changes to computer vision are making new applications that weren't possible a few years ago.
- Computer vision deep leaning techniques are always evolving making a new architectures which can help us in other
  areas other than computer vision.
- For example, Andrew Ng took some ideas of computer vision and applied it in speech recognition.
- Examples of a computer vision problems includes:
- Image classification.
  - Object detection.
    - Detect object and localize them.
  -Neural style transfer
    -Changes the style of an image using another image.
- One of the challenges of computer vision problem that images can be so large and we want a fast and accurate
algorithm to work with that
  - For example, a 1000x1000 image will represent 3 million feature/input to the full connected neural network. If the
    following hidden layer contains 1000, then we will want to learn weights of the shape [1000, 3 million] which is 3
    billion parameter only in the first layer and thats so computationally expensive!
 - One of the solutions is to build this using convolution layers instead of the fully connected layers.

# Edge detection example
- The convolution operation is one of the fundamentals blocks of a CNN. One of the examples about convolution is the
image edge detection operation.
- Early layers of CNN might detect edges then the middle layers will detect parts of objects and the later layers will put the
these parts together to produce an output.
- In an image we can detect vertical edges, horizontal edges, or full edge detector
- Vertical edge detection:
  - An example of convolution operation to detect vertical edges:
- 
