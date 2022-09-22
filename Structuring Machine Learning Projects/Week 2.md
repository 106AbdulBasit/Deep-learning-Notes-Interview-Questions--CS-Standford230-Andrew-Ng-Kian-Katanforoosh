# ML Strategy 2

# Carrying out error analysis

- Error analysis - process of manually examining mistakes that your algorithm is making. It can give you insights into what
to do next. E.g.:
    - In the cat classification example, if you have 10% error on your dev set and you want to decrease the error.
    - You discovered that some of the mislabeled data are dog pictures that look like cats. Should you try to make your
    - cat classifier do better on dogs (this could take some weeks)?
    - Error analysis approach:
        - Get 100 mislabeled dev set examples at random.
        - Count up how many are dogs.
        - if 5 of 100 are dogs then training your classifier to do better on dogs will decrease your error up to 9.5% (called
            ceiling), which can be too little.
        - if 50 of 100 are dogs then you could decrease your error up to 5%, which is reasonable and you should work on
          that.
 - Based on the last example, error analysis helps you to analyze the error before taking an action that could take lot of
time with no need.

- **table Missing update shortly**

- This quick counting procedure, which you can often do in, at most, small numbers of hours can really help you make
much better prioritization decisions, and understand how promising different approaches are to work on.



# Cleaning up incorrectly labeled data

- DL algorithms are quite robust to random errors in the training set but less robust to systematic errors. But it's OK to go
and fix these labels if you can.

-  **Table is missing need o update this later**
-  Consider these guidelines while correcting the dev/test mislabeled examples:
  - Apply the same process to your dev and test sets to make sure they continue to come from the same distribution.
  - Consider examining examples your algorithm got right as well as ones it got wrong. (Not always done if you
  - reached a good accuracy)
  - Train and (dev/test) data may now come from a slightly different distributions.
  - It's very important to have dev and test sets to come from the same distribution. But it could be OK for a train set to
  come from slightly other distribution.
  
  
  # Build your first system quickly, then iterate
  - The steps you take to make your deep learning project:
    - Setup dev/test set and metric
    - Build initial system quickly
    - Use Bias/Variance analysis & Error analysis to prioritize next steps.
  
  
  # Training and testing on different distributions
  - A lot of teams are working with deep learning applications that have training sets that are different from the dev/test
    sets due to the hunger of deep learning to data.
  - There are some strategies to follow up when training set distribution differs from dev/test sets distribution.
  - Option one (not recommended): shuffle all the data together and extract randomly training and dev/test sets.
    - Advantages: all the sets now come from the same distribution.
    - Disadvantages: the other (real world) distribution that was in the dev/test sets will occur less in the new
      dev/test sets and that might be not what you want to achieve.
  - Option two: take some of the dev/test set examples and add them to the training set.
    - Advantages: the distribution you care about is your target now.
    - Disadvantage: the distributions in training and dev/test sets are now different. But you will get a better
      performance over a long time.
      
  # Bias and Variance with mismatched data distributions
  - Bias and Variance analysis changes when training and Dev/test set is from the different distribution.
  -  Example: the cat classification example. Suppose you've worked in the example and reached this
      - Human error: 0%
      - Train error: 1% 
      - Dev error: 10%
      - In this example, you'll think that this is a variance problem, but because the distributions aren't the same you can't
        tell for sure. Because it could be that train set was easy to train on, but the dev set was more difficult.
  - To solve this issue we create a new set called train-dev set as a random subset of the training set (so it has the same
    distribution) and we get:
    - Human error: 0%
    - Train error: 1%
    - Train-dev error: 9%
    - Dev error: 10%
    - Now we are sure that this is a high variance problem.
  - Suppose we have a different situation:
    - Human error: 0%
    - Train error: 1%
    - Train-dev error: 1.5%
    - Dev error: 10%
    - In this case we have something called Data mismatch problem.
  - Conclusions:
    - i. Human-level error (proxy for Bayes error)
    - Train error
      - Calculate avoidable bias = training error - human level error
      - If the difference is big then its Avoidable bias problem then you should use a strategy for high bias.
   - Train-dev error
      - Calculate variance = training-dev error - training error
      - If the difference is big then its high variance problem then you should use a strategy for solving it.
   - Dev error
    - Calculate data mismatch = dev error - train-dev error
    - If difference is much bigger then train-dev error its Data mismatch problem
   - Test error
      - Calculate degree of overfitting to dev set = test error - dev error
      - Is the difference is big (positive) then maybe you need to find a bigger dev set (dev set and test set come from
        the same distribution, so the only way for there to be a huge gap here, for it to do much better on the dev set
         than the test set, is if you somehow managed to overfit the dev set).
   - 
