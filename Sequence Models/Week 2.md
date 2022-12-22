# Natural Language Processing & Word Embeddings
Natural language processing with deep learning is an important combination. Using word vector representations and
embedding layers you can train recurrent neural networks with outstanding performances in a wide variety of industries.
Examples of applications are sentiment analysis, named entity recognition and machine translation.

## Introduction to Word Embeddings

### Word Representation

- NLP has been revolutionized by deep learning and especially by RNNs and deep RNNs.
- Word embeddings is a way of representing words. It lets your algorithm automatically understand the analogies
between words like "king" and "queen".
- So far we have defined our language by a vocabulary. Then represented our words with a one-hot vector that represents
the word in the vocabulary.
  - An image example would be:
  - ![](![image](https://user-images.githubusercontent.com/36159918/209143893-e7269ddf-8b4d-4c18-9175-f2e9621b44c0.png))
  - We will use the annotation O for any word that is represented with one-hot like in the image.
  - One of the weaknesses of this representation is that it treats a word as a thing that itself and it doesn't allow an
    algorithm to generalize across words.
    - For example: "I want a glass of orange ______", a model should predict the next word as juice.
    - A similar example "I want a glass of apple ______", a model won't easily predict juice here if it wasn't trained on
      it. And if so the two examples aren't related although orange and apple are similar.
   - Inner product between any one-hot encoding vector is zero. Also, the distances between them are the same.
- So, instead of a one-hot presentation, won't it be nice if we can learn a featurized representation with each of these
   words: man, woman, king, queen, apple, and orange?
- ![](![image](https://user-images.githubusercontent.com/36159918/209144344-c3502f5f-b09b-4ddf-8f5e-5e0eee6b1e8c.png))
- Each word will have a, for example, 300 features with a type of float point number.
  - Each word column will be a 300-dimensional vector which will be the representation.
  - We will use the notation e5391 to describe man word features vector.
  - Now, if we return to the examples we described again:
    - "I want a glass of orange ______"
    - want a glass of apple ______
  - Orange and apple now share a lot of similar features which makes it easier for an algorithm to generalize between
    them.
  - We call this representation Word embeddings.
- To visualize word embeddings we use a t-SNE algorithm to reduce the features to 2 dimensions which makes it easy to
  visualize:
- ![](![image](https://user-images.githubusercontent.com/36159918/209144864-9e46756c-a6da-494b-a8c8-53c5cc7b135a.png))
  - You will get a sense that more related words are closer to each other.
- The word embeddings came from that we need to embed a unique vector inside a n-dimensional space.

### Using word embeddings
- Let's see how we can take the feature representation we have extracted from each word and apply it in the Named
entity recognition problem.
- Given this example (from named entity recognition):
- ![](![image](https://user-images.githubusercontent.com/36159918/209145534-620330b1-01b1-413e-a7ce-dbf453aab597.png))
- Sally Johnson is a person's name.
- After training on this sentence the model should find out that the sentence "Robert Lin is an apple farmer" contains
  Robert Lin as a name, as apple and orange have near representations.
- Now if you have tested your model with this sentence "Mahmoud Badry is a durian cultivator" the network should learn
  the name even if it hasn't seen the word durian before (during training). That's the power of word representations.
- The algorithms that are used to learn word embeddings can examine billions of words of unlabeled text - for example,
  100 billion words and learn the representation from them.
- Transfer learning and word embeddings:
  - i. Learn word embeddings from large text corpus (1-100 billion of words).
    - Or download pre-trained embedding online.
  - ii Transfer embedding to new task with the smaller training set (say, 100k words).
  - iii. Optional: continue to finetune the word embeddings with new data.
    - You bother doing this if your smaller training set (from step 2) is big enough.
  - Word embeddings tend to make the biggest difference when the task you're trying to carry out has a relatively smaller
    training set
  - Also, one of the advantages of using word embeddings is that it reduces the size of the input!
    - 10,000 one hot compared to 300 features vector.
  - Word embeddings have an interesting relationship to the face recognition task:
  - ![](![image](https://user-images.githubusercontent.com/36159918/209146467-630f1840-5fb1-4c20-8ac5-4b396b5246bf.png))
  - In this problem, we encode each face into a vector and then check how similar are these vectors.
  - Words encoding and embeddings have a similar meaning here
- In the word embeddings task, we are learning a representation for each word in our vocabulary (unlike in image
  encoding where we have to map each new image to some n-dimensional vector). We will discuss the algorithm in next
  sections.

### Properties of word embeddings

- One of the most fascinating properties of word embeddings is that they can also help with analogy reasoning. While
  analogy reasoning may not be by itself the most important NLP application, but it might help convey a sense of what
  these word embeddings can do.
- Analogies example:
  - Given this word embeddings table:
  - ![](![image](https://user-images.githubusercontent.com/36159918/209147030-c8fa80ac-d789-419a-9803-0cb9abb5d8a1.png))
- Can we conclude this relation:
  - Man ==> Woman
  - King ==> ??
- Lets subtract eMan from eWoman . This will equal the vector [-2 0 0 0]
- Similar eking - eQueen = [-2 0 0 0]
- So the difference is about the gender in both.
  - ![image](https://user-images.githubusercontent.com/36159918/209147468-9d7be30e-be30-4011-a688-c48709ba5519.png)
- This vector represents the gender.
- This drawing is a 2D visualization of the 4D vector that has been extracted by a t-SNE algorithm. It's a drawing
just for visualization. Don't rely on the t-SNE algorithm for finding parallels.

- So we can reformulate the problem to find:
  - eman - eWoman ≈ eking - e??
  - It can also be represented mathematically by:
  - ![](![image](https://user-images.githubusercontent.com/36159918/209147819-61e8795d-c766-4710-a69a-d0441b24a905.png)
  - It turns out that eQueen is the best solution here that gets the the similar vector

- Cosine similarity - the most commonly used similarity function:
  - Equation:
  - ![image](https://user-images.githubusercontent.com/36159918/209148007-686c81b3-3835-47f0-81a0-ef07ab869d2a.png)

  - CosineSimilarity(u, v) = u . v / ||u|| ||v|| = cos(θ)
  - The top part represents the inner product of u and v vectors. It will be large if the vectors are very similar
- You can also use Euclidean distance as a similarity function (but it rather measures a dissimilarity, so you should take it
  with negative sign).
  
- We can use this equation to calculate the similarities between word embeddings and on the analogy problem where u
- = eW and v = eking - eMan + eWoman


### Embedding matrix
- When you implement an algorithm to learn a word embedding, what you end up learning is a embedding matrix.
- Let's take an example:
  - Suppose we are using 10,000 words as our vocabulary (plus token).
  - The algorithm should create a matrix E of the shape (300, 10000) in case we are extracting 300 features.
  - ![](![image](https://user-images.githubusercontent.com/36159918/209148784-352886b5-1c6d-4746-8c49-2da7daa6fe07.png)
  - If O6257 is the one hot encoding of the word orange of shape (10000, 1), then
  - np.dot( E ,O6257 ) = e6257 which shape is (300, 1).
  - Generally np.dot( E , O ) =ej
- In the next sections, you will see that we first initialize E randomly and then try to learn all the parameters of this
  matrix.
- In practice it's not efficient to use a dot multiplication when you are trying to extract the embeddings of a specific word,
  instead, we will use slicing to slice a specific column. In Keras there is an embedding layer that extracts this column with
   no multiplication.
   


## Learning Word Embeddings: Word2vec & GloVe\

### Learning word embeddings

- Let's start learning some algorithms that can learn word embeddings.
- At the start, word embeddings algorithms were complex but then they got simpler and simpler.
- We will start by learning the complex examples to make more intuition.
- Neural language model:
  - Let's start with an example:
  - ![image](https://user-images.githubusercontent.com/36159918/209150694-ab93d4e6-d78a-482d-8cf9-efbbcbf4e036.png)
  - We want to build a language model so that we can predict the next word.
  - So we use this neural network to learn the language model
  - ![image](https://user-images.githubusercontent.com/36159918/209150993-32b0a026-29d2-4c70-adc0-bb20d0ce89ae.png)
  - We get e by np.dot( E ,o<sub>j</sub>)
  - NN layer has parameters W1 and b1 while softmax layer has parameters W2 and b2
  - Input dimension is (300*6, 1) if the window size is 6 (six previous words).
  - Here we are optimizing E matrix and layers parameters. We need to maximize the likelihood to predict the
    next word given the context (previous words).
- This model was build in 2003 and tends to work pretty decent for learning word embeddings.
- In the last example we took a window of 6 words that fall behind the word that we want to predict. There are other
  choices when we are trying to learn word embeddings.
- Suppose we have an example: "I want a glass of orange juice to go along with my cereal"
- To learn juice, choices of context are:
  - a. Last 4 words.
    - We use a window of last 4 words (4 is a hyperparameter), "a glass of orange" and try to predict the next
      word from it.
  - b 4 words on the left and on the right
    - "a glass of orange" and "to go along with"
  - c. Last 1 word.
    - "orange"
  - d. Nearby 1 word.
    - "glass" word is near juice.
    - This is the idea of skip grams model.
    - The idea is much simpler and works remarkably well.
    - We will talk about this in the next section.
- Researchers found that if you really want to build a language model, it's natural to use the last few words as a context.
  But if your main goal is really to learn a word embedding, then you can use all of these other contexts and they will result
  in very meaningful work embeddings as well.
- To summarize, the language modeling problem poses a machines learning problem where you input the context (like
  the last four words) and predict some target words. And posing that problem allows you to learn good word
  embeddings.
- Word2Vec
  - Before presenting Word2Vec, lets talk about **skip-grams**:
  - For example, we have the sentence: "I want a glass of orange juice to go along with my cereal"
  - We will choose context and target.
  - The target is chosen randomly based on a window with a specific size
  - ![word2vec](https://user-images.githubusercontent.com/36159918/209152916-d5244811-4820-4686-9f4e-c00f6ec8b449.PNG)
  - We have converted the problem into a supervised problem.
  - This is not an easy learning problem because learning within -10/+10 words (10 - an example) is hard
  - We want to learn this to get our word embeddings model.
 ## Word2Vec model:
  - Vocabulary size = 10,000 words
  - Let's say that the context word are c and the target word is t
  - We want to learn c to t
  - We get e by E . oc
  - We then use a softmax layer to get P(t|c) which is ŷ
  - Also we will use the cross-entropy loss function.
  - This model is called skip-grams model.
- The last model has a problem with the softmax layer:
   - ![image](https://user-images.githubusercontent.com/36159918/209153514-cccbf91b-66fe-4bf6-a963-b8b6ec87d964.png)
   - Here we are summing 10,000 numbers which corresponds to the number of words in our vocabulary.
   - If this number is larger say 1 million, the computation will become very slow.
   - One of the solutions for the last problem is to use "Hierarchical softmax classifier" which works as a tree classifier.
   - ![image](https://user-images.githubusercontent.com/36159918/209153740-de2dc402-fd0e-4ba9-a0a7-7c37d94516a0.png)

- In practice, the hierarchical softmax classifier doesn't use a balanced tree like the drawn one. Common words are at the
top and less common are at the bottom.
- How to sample the context c?
  - One way is to choose the context by random from your corpus.
  - If you have done it that way, there will be frequent words like "the, of, a, and, to, .." that can dominate other words
    like "orange, apple, durian,..."
  - In practice, we don't take the context uniformly random, instead there are some heuristics to balance the common
    words and the non-common words.
- word2vec paper includes 2 ideas of learning word embeddings. One is skip-gram model and another is CBoW
  (continuous bag-of-words).


## Negative Sampling
- Negative sampling allows you to do something similar to the skip-gram model, but with a much more efficient learning
algorithm. We will create a different learning problem.
- Given this example:
  - "I want a glass of orange juice to go along with my cereal"
- The sampling will look like this:
- ![NegativeSampling](https://user-images.githubusercontent.com/36159918/209156368-d1591848-3b2c-4889-8534-bef0eb339cdc.PNG)
- We get positive example by using the same skip-grams technique, with a fixed window that goes around.
- To generate a negative example, we pick a word randomly from the vocabulary.
- Notice, that we got word "of" as a negative example although it appeared in the same sentence.
- So the steps to generate the samples are:
  - i. Pick a positive context
  - ii. Pick a k negative contexts from the dictionary.
- k is recommended to be from 5 to 20 in small datasets. For larger ones - 2 to 5.
- We will have a ratio of k negative examples to 1 positive ones in the data we are collecting.
- Now let's define the model that will learn this supervised learning problem:
  - Lets say that the context word are c and the word are t and y is the target.
  - We will apply the simple logistic regression model.
  - ![image](https://user-images.githubusercontent.com/36159918/209156947-bd98437f-c68c-43a3-bce7-64d3903e1fdb.png)

 - The logistic regression model can be drawn like this:
 - ![image](https://user-images.githubusercontent.com/36159918/209156998-c7342b4f-87d3-4c31-a84b-3148e1b0a0cc.png)

 - So we are like having 10,000 binary classification problems, and we only train k+1 classifier of them in each
   iteration.
 - How to select negative samples:
 - We can sample according to empirical frequencies in words corpus which means according to how often different
  words appears. But the problem with that is that we will have more frequent words like the, of, and...
- The best is to sample with this equation (according to authors):
  - ![image](https://user-images.githubusercontent.com/36159918/209157231-dec91a71-8716-4096-aca9-ef27f91d6fb6.png)

### GloVe word vectors
- GloVe is another algorithm for learning the word embedding. It's the simplest of them.
- This is not used as much as word2vec or skip-gram models, but it has some enthusiasts because of its simplicity.
- GloVe stands for Global vectors for word representation.
- Let's use our previous example: "I want a glass of orange juice to go along with my cereal".
- We will choose a context and a target from the choices we have mentioned in the previous sections
- Then we will calculate this for every pair: X = # times t appears in context of c
- X = X if we choose a window pair, but they will not equal if we choose the previous words for example. In GloVe they
use a window which means they are equal
- The model is defined like this:
  - ![image](https://user-images.githubusercontent.com/36159918/209157622-6d958779-61c9-4e72-aadb-1405365a0745.png)

- f(x) - the weighting term, used for many reasons which include:
  - The log(0) problem, which might occur if there are no pairs for the given target and context values.
  - Giving not too much weight for stop words like "is", "the", and "this" which occur many times.
  - Giving not too little weight for infrequent words.
- Theta and e are symmetric which helps getting the final word embedding.
- Conclusions on word embeddings:
  - If this is your first try, you should try to download a pre-trained model that has been made and actually works best.
  - If you have enough data, you can try to implement one of the available algorithms.
  - Because word embeddings are very computationally expensive to train, most ML practitioners will load a pretrained
    set of embeddings.
  - A final note that you can't guarantee that the axis used to represent the features will be well-aligned with what
    might be easily humanly interpretable axis like gender, royal, age.
