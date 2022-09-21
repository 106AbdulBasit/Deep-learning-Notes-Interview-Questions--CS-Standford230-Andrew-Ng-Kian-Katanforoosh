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
 
 
