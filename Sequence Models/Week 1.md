# Recurrent Neural Networks

## Why sequence models
- Sequence Models like RNN and LSTMs have greatly transformed learning on sequences in the past few years.
- Examples of sequence data in applications:
  - Speech recognition (sequence to sequence):
  - X: wave sequence
  - Y: text sequence
-  Music generation (one to sequence):
  - X: nothing or an integer
  - Y: wave sequence
- Sentiment classification (sequence to one):
  - X: text sequence
  - Y: integer rating from one to five
- DNA sequence analysis (sequence to sequence):
 - X: DNA sequence
 - Y: DNA Labels
- Machine translation (sequence to sequence):
 - X: text sequence (in one language)
 - Y: text sequence (in other language)
- Video activity recognition (sequence to one):
  - X: video frames
  - Y: label (activity)
- Name entity recognition (sequence to sequence):
  - X: text sequence
  - Y: label sequence
  - Can be used by seach engines to index different type of words inside a text.
- All of these problems with different input and output (sequence or not) can be addressed as supervised learning with
label data X, Y as the training set.

# Notation
- In this section we will discuss the notations that we will use through the course.
- Motivating example:
  - Named entity recognition example:
  - Y: 1 1 0 1 1 0 0 0 0
  - Both elements has a shape of 9. 1 means its a name, while 0 means its not a name.
- We will index the first element of x by x , the second x and so on.
  - x = Harry
  - x = Potter
- Similarly, we will index the first element of y by y , the second y and so on.
  - y = 1
  - y = 1
- T is the size of the input sequence and T is the size of the output sequence.
  - T = T = 9 in the last example although they can be different in other problems.
- x is the element t of the sequence of input vector i. Similarly y means the t-th element in the output sequence
of the i training example.
- T the input sequence length for training example i. It can be different across the examples. Similarly for T will be the
length of the output sequence in the i-th training example.

- Representing words:
  - We will now work in this course with NLP which stands for natural language processing. One of the challenges of
NLP is how can we represent a word?
  - We need a vocabulary list that contains all the words in our target sets.
  - Example:
   - [a ... And ... Harry ... Potter ... Zulu]
   - Each word will have a unique index that it can be represented with.
   - The sorting here is in alphabetical order.
  - Vocabulary sizes in modern applications are from 30,000 to 50,000. 100,000 is not uncommon. Some of the
bigger companies use even a million.
  - To build vocabulary list, you can read all the texts you have and get m words with the most occurrence, or
search online for m most occurrent words.

- Create a one-hot encoding sequence for each word in your dataset given the vocabulary you have created
  - While converting, what if we meet a word thats not in your dictionary?
  - We can add a token in the vocabulary with name <UNK> which stands for unknown text and use its index for
your one-hot
- Full example:
  - 
