 # What does optimization algorith  do?
# Mini-batch gradient descent

- Training NN with a large data is slow. So to find an optimization algorithm that runs faster is a good idea.
-  Suppose we have m = 50 million . To train this data it will take a huge processing time for one step.  
      because 50 million won't fit in the memory at once we need other processing to make such a thing.
- It turns out you can make a faster algorithm to make gradient descent process some of your items even before you
    finish the 50 million items.
- Suppose we have split m to mini batches of size 1000.
  - X{1} = 0 ... 1000
  - X{2} = 1001 ... 2000
  - ...
  - X{bs} = ...
- We similarly split X & Y .
- So the definition of mini batches ==> t: X{t}, Y{t}
- In Batch gradient descent we run the gradient descent on the whole dataset.
  While in Mini-Batch gradient descent we run the gradient descent on the mini datasets.
- Mini-Batch algorithm pseudo code:
'''
for t = 1:No_of_batches # this is called an epoch
AL, caches = forward_prop(X{t}, Y{t})
cost = compute_cost(AL, Y{t})
grads = backward_prop(AL, caches)
update_parameters(grads)

'''

- The code inside an epoch should be vectorized.
- Mini-batch gradient descent works much faster in the large datasets.
- on 1 epoch it run for 5000 gradient
-  epoch is a word that means a single pass through the training set. Whereas with batch gradient descent, a single pass through the training set allows you to take      only one gradient descent step. With mini-batch gradient descent, a single pass through the training set, that is one epoch, allows you to take 5,000 gradient      descent steps. 
-  Now of course you want to take multiple passes through the training set which you usually want to, you might want another for loop for another while loop out there.

# Understanding mini-batch gradient descent
- In mini-batch algorithm, the cost won't go down with each step as it does in batch algorithm. It could contain some ups
and downs but generally it has to go down (unlike the batch gradient descent where cost function descreases on each

- ![image](https://user-images.githubusercontent.com/36159918/190920972-4a93a16f-ca8a-4743-a0f8-b6a9fb0bbebe.png)

- Mini-batch size:
      - ( mini batch size = m ) ==> Batch gradient descent
      - ( mini batch size = 1 ) ==> Stochastic gradient descent (SGD)
      - ( mini batch size = between 1 and m ) ==> Mini-batch gradient descent\
- Batch gradient descent:
      - too long per iteration (epoch)
- Stochastic gradient descent:
      -too noisy regarding cost minimization (can be reduced by using smaller learning rate)
      -won't ever converge (reach the minimum cost)
      -lose speedup from vectorization
-Mini-batch gradient descent:
      -i. faster learning:
      - you have the vectorization advantage
         make progress without waiting to process the entire training set
      - ii. doesn't always exactly converge (oscelates in a very small region, but you can reduce learning rate)
-     Guidelines for choosing mini-batch size:
      -     i. If small training set (< 2000 examples) - use batch gradient descent.
      -     ii. It has to be a power of 2 (because of the way computer memory is layed out and accessed, sometimes your code
      -     runs faster if your mini-batch size is a power of 2): 64, 128, 256, 512, 1024, ...
      -     iii. Make sure that mini-batch fits in CPU/GPU memory.
-Mini-batch size is a hyperparameter .

# Exponentially weighted averages
- There are optimization algorithms that are better than gradient descent, but you should first learn about Exponentially
weighted averages.
- If we have data like the temperature of day through the year it could be like this:
![image](https://user-images.githubusercontent.com/36159918/190921546-88493e3b-9366-431d-80df-b585cd0bf8cb.png)

t(1) = 40
t(2) = 49
t(3) = 45
...
t(180) =

- This data is small in winter and big in summer. If we plot this data we will find it some noisy.
- Now lets compute the Exponentially weighted averages:
- '''
    V0 = 0
V1 = 0.9 * V0 + 0.1 * t(1) = 4 # 0.9 and 0.1 are hyperparameters
V2 = 0.9 * V1 + 0.1 * t(2) = 8.5
V3 = 0.9 * V2 + 0.1 * t(3) = 12.15 
 '''
 
 -  General equation
- V(t) = beta * v(t-1) + (1-beta) * theta(t)
- If we plot this it will represent averages over ~ (1 / (1 - beta)) entries:
      - beta = 0.9 will average last 10 entries the red line
      - beta = 0.98 will average last 50 entries the green line
      - beta = 0.5 will average last 2 entries the yellow line

- Best beta average for our case is between 0.9 and 0.98
- Imagery

![image](https://user-images.githubusercontent.com/36159918/190921546-88493e3b-9366-431d-80df-b585cd0bf8cb.png)


# Understanding exponentially weighted averages
![image](https://user-images.githubusercontent.com/36159918/190922115-7e08debc-c15e-48a8-8768-96268d2e32e1.png)

- We can implement this algorithm with more accurate results using a moving window. But the code is more efficient and
faster using the exponentially weighted averages algorithm.
- Algorithm is very simple:
'''
v = 0
Repeat
{
Get theta(t)
v = beta * v + (1-beta) * theta(t)
}
'''

# Bias correction in exponentially weighted averages

- The bias correction helps make the exponentially weighted averages more accurate.
- Because v(0) = 0 , the bias of the weighted averages is shifted and the accuracy suffers at the start.
- To solve the bias issue we have to use this equation:
     v(t) = (beta * v(t-1) + (1-beta) * theta(t)) / (1 - beta^t)
- As t becomes larger the (1 - beta^t) becomes close to 1

# Gradient descent with momentum

- The momentum algorithm almost always works faster than standard gradient descent.
- The simple idea is to calculate the exponentially weighted averages for your gradients and then update your weights
      with the new values.
- Pseudo code:
-     '''
vdW = 0, vdb = 0
on iteration t:
# can be mini-batch or batch gradient descent
compute dw, db on current mini-batch
vdW = beta * vdW + (1 - beta) * dW
vdb = beta * vdb + (1 - beta) * db
W = W - learning_rate * vdW
b = b - learning_rate * vdb
'''

- Momentum helps the cost function to go to the minimum point in a more fast and consistent way.
- beta is another hyperparameter . beta = 0.9 is very common and works very well in most cases.
- In practice people don't bother implementing bias correction.

# RMSprop
- Stands for Root mean square prop.
- This algorithm speeds up the gradient descent.
- Pseudo code:
      '''
      sdW = 0, sdb = 0
on iteration t:
 can be mini-batch or batch gradient descent
compute dw, db on current mini-batch
sdW = (beta * sdW) + (1 - beta) * dW^2 # squaring is element-wise
sdb = (beta * sdb) + (1 - beta) * db^2 # squaring is element-wise
W = W - learning_rate * dW / sqrt(sdW)
b = B - learning_rate * db / sqrt(sdb)
      '''
      
- RMSprop will make the cost function move slower on the vertical direction and faster on the horizontal direction in the
following example:
- ![image](https://user-images.githubusercontent.com/36159918/190924157-cd6c2ca0-5a03-4a47-b7d5-e2d6581406a4.png)
 - b = vertical line the up direction and w is a horizontal line the forwarad direction we have to minimize the distance

- Ensure that sdW is not zero by adding a small value epsilon (e.g. epsilon = 10^-8 ) to it:
W = W - learning_rate * dW / (sqrt(sdW) + epsilon)

- With RMSprop you can increase your learning rate.
- Developed by Geoffrey Hinton and firstly introduced on Coursera.org course.

# Adam optimization algorithm
- Stands for Adaptive Moment Estimation.
- Adam optimization and RMSprop are among the optimization algorithms that worked very well with a lot of NN
architectures.
- Pseudo code:
      '''
      vdW = 0, vdW = 0
sdW = 0, sdb = 0
on iteration t:
# can be mini-batch or batch gradient descent
compute dw, db on current mini-batch
vdW = (beta1 * vdW) + (1 - beta1) * dW # momentum
vdb = (beta1 * vdb) + (1 - beta1) * db # momentum
sdW = (beta2 * sdW) + (1 - beta2) * dW^2 # RMSprop
sdb = (beta2 * sdb) + (1 - beta2) * db^2 # RMSprop
vdW = vdW / (1 - beta1^t) # fixing bias
vdb = vdb / (1 - beta1^t) # fixing bias
sdW = sdW / (1 - beta2^t) # fixing bias
sdb = sdb / (1 - beta2^t) # fixing bias
W = W - learning_rate * vdW / (sqrt(sdW) + epsilon)
b = B - learning_rate * vdb / (sqrt(sdb) + epsilon)
  
  '''
 - Hyperparameters for Adam:
      - Learning rate: needed to be tuned.
      - beta1 : parameter of the momentum - 0.9 is recommended by default.
      - beta2 : parameter of the RMSprop - 0.999 is recommended by default.
      - epsilon : 10^-8 is recommended by default
 - adam was presented by [Diederik Kingma](http://dpkingma.com/)

# Learning rate decay
- Slowly reduce learning rate.
- As mentioned before mini-batch gradient descent won't reach the optimum point (converge). But by making the
learning rate decay with iterations it will be much closer to it because the steps (and possible oscillations) near the
optimum are smaller
- One technique equations is learning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate_0
      - epoch_num is over all data (not a single mini-batch).
- Other learning rate decay methods (continuous):
      - learning_rate = (0.95 ^ epoch_num) * learning_rate_0
      - learning_rate = (k / sqrt(epoch_num)) * learning_rate_0
- Some people perform learning rate decay discretely - repeatedly decrease after some number of epochs.
- Some people are making changes to the learning rate manually.
- decay_rate is another hyperparameter .
- For Andrew Ng, learning rate decay has less priority

# The problem of local optima
- The normal local optima is not likely to appear in a deep neural network because data is usually high dimensional. For
point to be a local optima it has to be a local optima for each of the dimensions which is highly unlikely.
- It's unlikely to get stuck in a bad local optima in high dimensions, it is much more likely to get to the saddle point rather
to the local optima, which is not a problem

![Saddle](https://raw.githubusercontent.com/106AbdulBasit/Deep-learning-Notes-Interview-Questions--CS-Standford230-Andrew-Ng-Kian-Katanforoosh/main/Images/saddle.PNG)
- Plateaus can make learning slow:
      - Plateau is a region where the derivative is close to zero for a long time.
      - This is where algorithms like momentum, RMSprop or Adam can help.
