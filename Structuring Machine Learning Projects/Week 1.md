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


# When to change dev/test sets and metrics

- Let's take an example. In a cat classification example we have these metric results:

| Metric  | Classification error |
| ------------- | ------------- |
| Algorithm A  | 3% error (But a lot of porn images are treated as cat images here)  |
| Algorithm B  | 5 % error  |

  - In the last example if we choose the best algorithm by metric it would be "A", but if the users decide it will be "B"
  - Thus in this case, we want and need to change our metric.
  - OldMetric = (1/m) * sum(y_pred[i] != y[i] ,m)
    - Where m is the number of Dev set items.
  - NewMetric = (1/sum(w[i])) * sum(w[i] * (y_pred[i] != y[i]) ,m)
    - where:
      - w[i] = 1 if x[i] is not porn
      - w[i] = 10 if x[i] is porn
  - This is actually an example of an orthogonalization where you should take a machine learning problem and break it into
distinct steps:
  - Figure out how to define a metric that captures what you want to do - place the target.
  - Worry about how to actually do well on this metric - how to aim/shoot accurately at the target.
- Conclusion: if doing well on your metric + dev/test set doesn't correspond to doing well in your application, change
your metric and/or dev/test set.


# Why human-level performance?

- We compare to human-level performance because of two main reasons:
  - Because of advances in deep learning, machine learning algorithms are suddenly working much better and so it has
    become much more feasible in a lot of application areas for machine learning algorithms to actually become
    competitive with human-level performance.
  - ii. It turns out that the workflow of designing and building a machine learning system is much more efficient when
    you're trying to do something that humans can also do.
- After an algorithm reaches the human level performance the progress and accuracy slow down.
  - ![image](https://user-images.githubusercontent.com/36159918/191527122-a0c4aa99-35d5-4deb-81e0-cd08d0864a44.png)
- You won't surpass an error that's called "Bayes optimal error".
- There isn't much error range between human-level error and Bayes optimal error.
- Humans are quite good at a lot of tasks. So as long as Machine learning is worse than humans, you can:
  - Get labeled data from humans.
  -  Gain insight from manual error analysis: why did a person get it right?
  -  Better analysis of bias/variance.
# Avoidable bias
- Suppose that the cat classification algorithm gives these results:
 Humans | 1 | 7.5  |
| ------------- | ------------- |  ------------- |
| Training error  | 8  |  8  |
| Dev Error | 10  |  10  |

  - In the left example, because the human level error is 1% then we have to focus on the bias.
  - In the right example, because the human level error is 7.5% then we have to focus on the variance.
  - The human-level error as a proxy (estimate) for Bayes optimal error. Bayes optimal error is always less (better), but
  - human-level in most cases is not far from it.
  - You can't do better than Bayes error unless you are overfitting.
  - Avoidable bias = Training error - Human (Bayes) error
  - Variance = Dev error - Training error

# Understanding human-level performance

- When choosing human-level performance, it has to be chosen in the terms of what you want to achieve with the system.
- You might have multiple human-level performances based on the human experience. Then you choose the human-level
- performance (proxy for Bayes error) that is more suitable for the system you're trying to build.
- Improving deep learning algorithms is harder once you reach a human-level performance.
- Summary of bias/variance with human-level performance:
  - human-level error (proxy for Bayes error)
    - Calculate avoidable bias = training error - human-level error
    - If avoidable bias difference is the bigger, then it's bias problem and you should use a strategy for bias
       resolving.
  - training error
    - Calculate variance = dev error - training error
    - If variance difference is bigger, then you should use a strategy for variance resolving.
  - Dev error
- So having an estimate of human-level performance gives you an estimate of Bayes error. And this allows you to more
  quickly make decisions as to whether you should focus on trying to reduce a bias or trying to reduce the variance of
  your algorithm.
- These techniques will tend to work well until you surpass human-level performance, whereupon you might no longer
  have a good estimate of Bayes error that still helps you make this decision really clearly.
  
  
 # Surpassing human-level performance
 
 - In some problems, deep learning has surpassed human-level performance. Like:
  - Online advertising.
  - Product recommendation.
  - Loan approval.
 - The last examples are not natural perception task, rather learning on structural data. Humans are far better in natural
perception tasks like computer vision and speech recognition.
 - It's harder for machines to surpass human-level performance in natural perception task. But there are already some
systems that achieved it.

# Improving your model performance
- The two fundamental asssumptions of supervised learning:
  - You can fit the training set pretty well. This is roughly saying that you can achieve low avoidable bias.
  - The training set performance generalizes pretty well to the dev/test set. This is roughly saying that variance is not
  too bad.
  
- To improve your deep learning supervised system follow these guidelines:
  - Look at the difference between human level error and the training error - avoidable bias.
  - ii. Look at the difference between the dev/test set and training set error - Variance.
- If avoidable bias is large you have these options
  - Train bigger model.
  - Train longer/better optimization algorithm (like Momentum, RMSprop, Adam).
  - Find better NN architecture/hyperparameters search.
- If variance is large you have these options:
  - Get more training data.
  - Regularization (L2, Dropout, data augmentation).
  - Find better NN architecture/hyperparameters search.
