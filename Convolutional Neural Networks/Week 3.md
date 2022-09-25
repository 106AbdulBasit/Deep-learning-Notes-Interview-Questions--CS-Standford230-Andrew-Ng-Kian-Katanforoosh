# Object Localization

- Object detection is one of the areas in which deep learning is doing great in the past two years.
- What are localization and detection?
  - Image Classification
    - Classify an image to a specific class. The whole image represents one class. We don't want to know exactly
     where are the object. Usually only one object is presented.
     ![image](https://user-images.githubusercontent.com/36159918/192136880-b80d5608-3a11-4bb7-baca-c217ef27651d.png)
  - Classification with localization:
    - Given an image we want to learn the class of the image and where are the class location in the image. We need
      to detect a class and a rectangle of where that object is. Usually only one object is presented.
      
      ![image](https://user-images.githubusercontent.com/36159918/192136915-df37456b-1024-41fe-b01c-fdde31467d04.png)
 - Semantic Segmentation:
    - We want to Label each pixel in the image with a category label. Semantic Segmentation Don't differentiate
      instances, only care about pixels. It detects no objects just pixels.
    - If there are two objects of the same class is intersected, we won't be able to separate them.
      ![image](https://user-images.githubusercontent.com/36159918/192136938-45da402e-df3e-4038-af56-f25fa5e1c719.png)

  - Instance Segmentation
    - This is like the full problem. Rather than we want to predict the bounding box, we want to know which pixel
      label but also distinguish them.
    - ![image](https://user-images.githubusercontent.com/36159918/192136979-04bffbff-f72d-4b94-8c41-1fced115ac2d.png)
  - To make image classification we use a Conv Net with a Softmax attached to the end of it.
  - To make classification with localization we use a Conv Net with a softmax attached to the end of it and a four numbers
     bx , by , bh , and bw to tell you the location of the class in the image. The dataset should contain this four numbers
      with the class too.
  - Defining the target label Y in classification with localization problem:

         Y = [
            Pc # Probability of an object is presented
            bx # Bounding box
            by # Bounding box
            bh # Bounding box
            bw # Bounding box
          c1 # The classes
          c2
          ...
          ]
- Example (Object is present):
  -       Y = [
            1 # Object is present
              0
              0
              100
              100
               0
              1
              0
            ]
- Example (When object isn't presented):
  -   = [
        0 # Object isn't presented
        ? # ? means we dont care with other values
          ?
          ?
          ?
          ?
          ?
          ?
        ]
  - The loss function for the Y we have created (Example of the square error):
    -  L(y',y) = {
              (y1'-y1)^2 + (y2'-y2)^2 + ... if y1 = 1
              (y1'-y1)^2 if y1 = 0
              }
    - In practice we use logistic regression for pc , log likely hood loss for classes, and squared error for the bounding
      box.
  
  
# Land mark Detection
- In some of the computer vision problems you will need to output some points. That is called landmark detection.
- For example, if you are working in a face recognition problem you might want some points on the face like corners of
   the eyes, corners of the mouth, and corners of the nose and so on. This can help in a lot of application like detecting the
   pose of the face.
- Y shape for the face recognition problem that needs to output 64 landmarks:
  Y = [
      THereIsAface # Probability of face is presented 0 or 1
        l1x,
        l1y,
        ....,
        l64x,
        l64y
          ]
- Another application is when you need to get the skeleton of the person using different landmarks/points in the person
which helps in some applications.
- Hint, in your labeled data, if l1x,l1y is the left corner of left eye, all other l1x,l1y of the other examples has to be the
  same.
  
 
# Object Detection

- We will use a Conv net to solve the object detection problem using a technique called the sliding windows detection
 algorithm.
- For example lets say we are working on Car object detection.
- The first thing, we will train a Conv net on cropped car images and non car images
  ![image](https://user-images.githubusercontent.com/36159918/192137613-7b2e25b9-c2d9-4854-b136-2595ac0cdcf8.png)
- After we finish training of this Conv net we will then use it with the sliding windows technique.
- Sliding windows detection algorithm:
  - i. Decide a rectangle size.
    ii. Split your image into rectangles of the size you picked. Each region should be covered. You can use some strides.
    iii. For each rectangle feed the image into the Conv net and decide if its a car or not.
    iv. Pick larger/smaller rectangles and repeat the process from 2 to 3.
    v. Store the rectangles that contains the cars.
    vi. If two or more rectangles intersects choose the rectangle with the best accuracy.
- Disadvantage of sliding window is the computation time.
- In the era of machine learning before deep learning, people used a hand crafted linear classifiers that classifies the
  object and then use the sliding window technique. The linear classier make it a cheap computation. But in the deep 
    learning era that is so computational expensive due to the complexity of the deep learning model.
- To solve this problem, we can implement the sliding windows with a Convolutional approach.
- One other idea is to compress your deep learning model

# Convolutional Implementation of Sliding Windows
- Turning FC layer into convolutional layers (predict image class from four classes):
- ![image](https://user-images.githubusercontent.com/36159918/192138129-6a8bb1e5-962e-4abf-b13c-8b335eebd0c7.png)
- As you can see in the above image, we turned the FC layer into a Conv layer using a convolution with the width and
height of the filter is the same as the width and height of the input.
- Convolution implementation of sliding windows:
  - First lets consider that the Conv net you trained is like this (No FC all is conv layers):
  - ![image](https://user-images.githubusercontent.com/36159918/192138154-78b81e9a-d4c9-4899-b805-c678cf899cc0.png)

  - Say now we have a 16 x 16 x 3 image that we need to apply the sliding windows in. By the normal implementation
    that have been mentioned in the section before this, we would run this Conv net four times each rectangle size will
    be 16 x 16.
    
  - The convolution implementation will be as follows
  - ![image](https://user-images.githubusercontent.com/36159918/192138179-bf8b39f0-476f-4436-9769-965e63cf9405.png)

  - Simply we have feed the image into the same Conv net we have trained.
  - The left cell of the result "The blue one" will represent the the first sliding window of the normal implementation.
  - The other cells will represent the others.
  - Its more efficient because it now shares the computations of the four times needed.
  - Another example would be:
    ![image](https://user-images.githubusercontent.com/36159918/192138209-84c0c9ae-a53a-4a4e-8a49-fe729fcf7f0a.png)
  - This example has a total of 16 sliding windows that shares the computation together.
  - ![](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Convolution%20sliding%20window.PNG)
  - Let's say this is 14 by 14 and run that through your convnet and do that for the next region over, then do that for the next 14 by 14 region, then the next one,    then the next one, then the next one, then the next one and so on, until hopefully that one recognizes the car. But now, instead of doing it sequentially, with     this convolutional implementation that you saw in the previous slide, you can implement the entire image, all maybe 28 by 28 and convolutionally make all the         predictions at the same time by one forward pass through this big convnet and hopefully have it recognize the position of the car.
  - The weakness of the algorithm is that the position of the rectangle wont be so accurate. Maybe none of the rectangles is
    exactly on the object you want to recognize.
 
 # Bounding Box Predictions
-  A better algorithm than the one described in the last section is the YOLO algorithm.
- YOLO stands for you only look once and was developed back in 2015.
- Yolo Algorithm:
  - ![Yolo](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Yoloalgorithm.PNG)
  - Lets say we have an image of 100 X 100
  - Place a 3 x 3 grid on the image. For more smother results you should use 19 x 19 for the 100 x 100
  - iii. Apply the classification and localization algorithm we discussed in a previous section to each section of the grid.
    bx and by will represent the center point of the object in each grid and will be relative to the box so the range is
      between 0 and 1 while bh and bw will represent the height and width of the object which can be greater than 1.0
      but still a floating point value.
  - iv. Do everything at once with the convolution sliding window. If Y shape is 1 x 8 as we discussed before then the
      output of the 100 x 100 image should be 3 x 3 x 8 which corresponds to 9 cell results.
  - v. Merging the results using predicted localization mid point.
  - We have a problem if we have found more than one object in one grid box.
- One of the best advantages that makes the YOLO algorithm popular is that it has a great speed and a Conv net
 implementation.
- How is YOLO different from other Object detectors? YOLO uses a single CNN network for both classification and
- localizing the object using bounding boxes.
- In the next sections we will see some ideas that can make the YOLO algorithm better.



