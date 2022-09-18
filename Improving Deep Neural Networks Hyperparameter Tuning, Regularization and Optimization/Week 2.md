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
