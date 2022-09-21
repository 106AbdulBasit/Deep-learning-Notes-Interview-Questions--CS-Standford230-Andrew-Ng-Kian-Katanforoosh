# ML Strategy 1
# Why ML Strategy

- You have a lot of ideas for how to improve the accuracy of your deep learning system:
  - Collect more data.
  - Collect more diverse training set.
  - Train algorithm longer with gradient descent.
  - Try different optimization algorithm (e.g. Adam).
  - Try bigger network.
  - Try smaller network.
  - Try dropout.
  - Add L2 regularization.
  -Change network architecture (activation functions, # of hidden units, etc.)
  

# Orthogonalization

- Some deep learning developers know exactly what hyperparameter to tune in order to try to achieve one effect. This is a
process we call orthogonalization.
- In orthogonalization, you have some controls, but each control does a specific task and doesn't affect other controls.
- For a supervised learning system to do well, you usually need to tune the knobs of your system to make sure that four
things hold true - chain of assumptions in machine learning:
    - i You'll have to fit training set well on cost function (near human level performance if possible).
        - If it's not achieved you could try bigger network, another optimization algorithm (like Adam)...
    - ii. Fit dev set well on cost function.
        - If its not achieved you could try regularization, bigger training set...
    - iii. Fit test set well on cost function.
        - If its not achieved you could try bigger dev. set...
    - iv. Performs well in real world.
      - If its not achieved you could try change dev. set, change cost function...
 
 
# Single number evaluation metric
- Its better and faster to set a single number evaluation metric for your project before you start it.
- Difference between precision and recall (in cat classification example):
  - Suppose we run the classifier on 10 images which are 5 cats and 5 non-cats. The classifier identifies that there are 4
cats, but it identified 1 wrong cat
  - Confusion matrix:
    
    Predicted        cat     Predicted non-cat
      Actual cat      3        2
      Actual non-cat   1        4
      
      
    - Precision: percentage of true cats in the recognized result: P = 3/(3 + 1) (of examples recognize as a cat  what percentage  acctually  are cats)
    - Recall: percentage of true recognition cat of the all cat predictions: R = 3/(3 + 2) (What percentage of actual cats are correctly recognized)
    -  Accuracy: (3+4)/10


- Using a precision/recall for evaluation is good in a lot of cases, but separately they don't tell you which algothims is
better. Ex:

Classifier Precision   Recall
  A           95%      90%
  B            98%       85%
  
  - A better thing is to combine precision and recall in one single (real) number evaluation metric. There a metric called F1
    score, which combines them
    
   - You can think of F1 score as an average of precision and recall F1 = 2 / ((1/P) + (1/R)) (Known as harmonic mean)


# Satisfying and Optimizing metric

- Its hard sometimes to get a single number evaluation metric. Ex:
 Classifier   F1     Running time
   A         90%       80 ms
   B         92%       95 ms
   C        92%      1,500 ms
   
 - So we can solve that by choosing a single optimizing metric and decide that other metrics are satisfying. Ex:
  - Maximize F1 # optimizing metric
    subject to running time < 100ms # satisficing metric
 - So as a general rule:
    - Maximize 1 # optimizing metric (one optimizing metric) 
      subject to N-1 # satisficing metric (N-1 satisficing metrics)
      
 # Train/dev/test distributions
 - Dev and test sets have to come from the same distribution.
 - Choose dev set and test set to reflect data you expect to get in the future and consider important to do well on.
 - Setting up the dev set, as well as the validation metric is really defining what target you want to aim at.
 
 # Size of the dev and test sets
 - An old way of splitting the data was 70% training, 30% test or 60% training, 20% dev, 20% test.
 - The old way was valid for a number of examples ~ <100000
 - In the modern deep learning if you have a million or more examples a reasonable split would be 98% training, 1% dev,
   1% test.
