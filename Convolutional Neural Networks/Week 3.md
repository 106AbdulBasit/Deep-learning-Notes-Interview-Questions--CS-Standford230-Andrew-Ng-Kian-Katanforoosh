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


# Intersection Over Union
- Intersection Over Union is a function used to evaluate the object detection algorithm.
- It computes size of intersection and divide it by the union. More generally, IoU is a measure of the overlap between two
  bounding boxes.
- For example:
  ![image](https://user-images.githubusercontent.com/36159918/192140001-2348a7d4-156b-484e-b31f-48880c51516f.png)
  - The red is the labeled output and the purple is the predicted output.
  - To compute Intersection Over Union we first compute the union area of the two rectangles which is "the first
    rectangle + second rectangle" Then compute the intersection area between these two rectangles.
  - Finally IOU = intersection area / Union area
 - If IOU >=0.5 then its good. The best answer will be 1.
 - The higher the IOU the better is the accuracy.


# Non-max Suppression
- One of the problems we have addressed in YOLO is that it can detect an object multiple times.
- Non-max Suppression is a way to make sure that YOLO detects the object just once.
 For example:
  ![image](https://user-images.githubusercontent.com/36159918/192140057-028f4b21-243e-4c93-b365-4cb26924d43b.png)
- Each car has two or more detections with different probabilities. This came from some of the grids that thinks that
 this is the center point of the object.
 
 - Non-max suppression algorithm:
  - i. Lets assume that we are targeting one class as an output class.
  - ii. Y shape should be [Pc, bx, by, bh, hw] Where Pc is the probability if that object occurs.
  - iii. Discard all boxes with Pc < 0.6
  - iv. While there are any remaining boxes:
  - Pick the box with the largest Pc Output that as a prediction.
    b. Discard any remaining box with IoU > 0.5 with that box output in the previous step i.e any box with high
    overlap(greater than overlap threshold of 0.5).
  - If there are multiple classes/object types c you want to detect, you should run the Non-max suppression c times,
once for every output class.

**Question how does the box made when we have 19*19 grid** 


# Anchor Boxes

- In YOLO, a grid only detects one object. What if a grid cell wants to detect multiple object?
  ![image](https://user-images.githubusercontent.com/36159918/192140148-907fcfcb-7a90-4da0-a375-efa669185989.png)
- Car and person grid is same here.
- In practice this happens rarely

- The idea of Anchor boxes helps us solving this issue.
- If Y = [Pc, bx, by, bh, bw, c1, c2, c3] Then to use two anchor boxes like this:
  - Y = [Pc, bx, by, bh, bw, c1, c2, c3, Pc, bx, by, bh, bw, c1, c2, c3] We simply have repeated the one anchor
    Y.
  - The two anchor boxes you choose should be known as a shape:
  - ![image](https://user-images.githubusercontent.com/36159918/192140192-cdf2e379-5e22-484d-bbb4-7f4700388284.png)

- So Previously, each object in training image is assigned to grid cell that contains that object's midpoint.
- With two anchor boxes, Each object in training image is assigned to grid cell that contains object's midpoint and anchor
  box for the grid cell with highest IoU. You have to check where your object should be based on its rectangle closest to
  which anchor box.
- Example of data:
  - ![image](https://user-images.githubusercontent.com/36159918/192140224-0c20b983-463a-486e-8d8f-0318b98a3554.png)

  - Where the car was near the anchor 2 than anchor 1.
  - You may have two or more anchor boxes but you should know their shapes.
- how do you choose the anchor boxes and people used to just choose them by hand. Maybe five or ten anchor box
  shapes that spans a variety of shapes that cover the types of objects you seem to detect frequently.
  You may also use a k-means algorithm on your dataset to specify that.
- Anchor boxes allows your algorithm to specialize, means in our case to easily detect wider images or taller ones.




 # YOLO Algorithm
-  YOLO is a state-of-the-art object detection model that is fast and accurate
- Lets sum up and introduce the whole YOLO algorithm given an example.
- Suppose we need to do object detection for our autonomous driver system.It needs to identify three classes:
  - i. Pedestrian (Walks on ground).
  - ii. Car.
  - iii. Motorcycle.
- We decided to choose two anchor boxes, a taller one and a wide one.
  - Like we said in practice they use five or more anchor boxes hand made or generated using k-means.
- Our labeled Y shape will be [Ny, HeightOfGrid, WidthOfGrid, 16] , where Ny is number of instances and each row (of
  size 16) is as follows:
  - [Pc, bx, by, bh, bw, c1, c2, c3, Pc, bx, by, bh, bw, c1, c2, c3]
- Your dataset could be an image with a multiple labels and a rectangle for each label, we should go to your dataset and
make the shape and values of Y like we agreed.
- An example:
  - ![image](https://user-images.githubusercontent.com/36159918/192146209-c9c6608c-d900-4b0e-adc3-306402b8ac17.png)

- We first initialize all of them to zeros and ?, then for each label and rectangle choose its closest grid point then the
shape to fill it and then the best anchor point based on the IOU. so that the shape of Y for one image should be
[HeightOfGrid, WidthOfGrid,16]
- Train the labeled images on a Conv net. you should receive an output of [HeightOfGrid, WidthOfGrid,16] for our case.
- To make predictions, run the Conv net on an image and run Non-max suppression algorithm for each class you have in
  our case there are 3 classes.
- You could get something like that:
  - ![image](https://user-images.githubusercontent.com/36159918/192146250-8ef523d7-72f2-426d-b5f6-0ea245256a8e.png)

  - Total number of generated boxes are grid_width * grid_height * no_of_anchors = 3 x 3 x 2
- By removing the low probability predictions you should have:
  - ![image](https://user-images.githubusercontent.com/36159918/192146278-6aeb84a4-ba5e-4445-a703-fe0f7209e4c8.png)

  - Then get the best probability followed by the IOU filtering:
    - ![image](https://user-images.githubusercontent.com/36159918/192146295-29675082-ac37-4df8-96c3-61fc4121a0cd.png)

- YOLO9000 Better, faster, stronger
  Summary
  - You can find implementations for YOLO here:
    - [https://github.com/allanzelener/YAD2K](https://github.com/allanzelener/YAD2K)
    - [https://github.com/thtrieu/darkflow](https://github.com/thtrieu/darkflow)
    - [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)


# Region Proposals (R-CNN)
- R-CNN is an algorithm that also makes an object detection.
- Yolo tells that its faster:
  - Our model has several advantages over classifier-based systems. It looks at the whole image at test time so its
    predictions are informed by global context in the image. It also makes predictions with a single network
    evaluation unlike systems like R-CNN which require thousands for a single image. This makes it extremely fast,
    more than 1000x faster than R-CNN and 100x faster than Fast R-CNN. See our paper for more details on the
    full system.
- But one of the downsides of YOLO that it process a lot of areas where no objects are present.
- R-CNN stands for regions with Conv Nets.
- R-CNN tries to pick a few windows and run a Conv net (your confident classifier) on top of them.
- The algorithm R-CNN uses to pick windows is called a segmentation algorithm. Outputs something like this:
  - ![image](https://user-images.githubusercontent.com/36159918/192146834-c34fdfad-965d-4b07-9235-0c814576cce9.png)

- If for example the segmentation algorithm produces 2000 blob then we should run our classifier/CNN on top of these
  blobs.
-  There has been a lot of work regarding R-CNN tries to make it faster:
  - R-CNN
    - Propose regions. Classify proposed regions one at a time. Output label + bounding box.
    - Downside is that its slow.
  - Fast R-CNN:
    - Propose regions. Use convolution implementation of sliding windows to classify all the proposed regions.
  - Faster R-CNN:
    - Use convolutional network to propose regions.
  - Mask RCN
- Most of the implementation of faster R-CNN are still slower than YOLO.
- Andrew Ng thinks that the idea behind YOLO is better than R-CNN because you are able to do all the things in just one
   time instead of two times.
- Other algorithms that uses one shot to get the output includes SSD and MultiBox.
- R-FCN is similar to Faster R-CNN but more efficient.

# Semantic Segmentation with U-Net
![obvssm](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Object%20Detection%20vs%20semenantic.PNG)
- an object detection algorithm, the goal may be to draw bounding boxes like these around the other vehicles
- learning algorithm to figure out what is every single pixel in this image, then you may use a semantic segmentation algorithm

Motivation for Unet
- ![Motivatio](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/MotivationforUnet.PNG)
- This segmentation can make it easier to spot irregularities and diagnose serious diseases and also help surgeons with planning out surgeries
-  learning algorithm can segment out the tumor automatically; this saves radiologists 

**Pixel Per Pixel Class Label*

-job of the segmentation algorithm of the unit algorithm will be to output, either one or zero for every single pixel in this image 
- If the car then pixel = 1
- if the not car  then pixel = 0
- If you have more class then more class labels
- 1[PP](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/Perpixelclasslabel.PNG)






