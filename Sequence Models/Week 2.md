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
- 
)
