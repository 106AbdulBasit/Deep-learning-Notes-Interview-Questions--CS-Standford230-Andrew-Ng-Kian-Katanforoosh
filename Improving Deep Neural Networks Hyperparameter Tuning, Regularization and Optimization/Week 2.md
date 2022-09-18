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
