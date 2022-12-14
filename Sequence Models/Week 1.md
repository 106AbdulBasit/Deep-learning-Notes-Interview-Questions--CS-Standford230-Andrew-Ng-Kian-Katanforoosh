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
  - ![image](https://user-images.githubusercontent.com/36159918/206914302-aee96bea-20bd-4dab-87b1-95e6065bc157.png)
  - The goal is given this representation for x to learn a mapping using a sequence model to then target output y as a
supervised learning problem
- Why not to use a standard network for sequence tasks? There are two problems:
  - Inputs, outputs can be different lengths in different examples.
  - This can be solved for normal NNs by paddings with the maximum lengths but it's not a good solution.
- Doesn't share features learned across different positions of text/sequence.
  - Using a feature sharing like in CNNs can significantly reduce the number of parameters in your model. That's
    what we will do in RNNs.
- Recurrent neural network doesn't have either of the two mentioned problems.
- Lets build a RNN that solves name entity recognition task:
  - ![image](https://user-images.githubusercontent.com/36159918/206914391-ec1e1473-e9ed-48a3-903a-e4509dd56e91.png)
      - In this problem T = T . In other problems where they aren't equal, the RNN architecture may be different.
      - a is usually initialized with zeros, but some others may initialize it randomly in some cases.
      - There are three weight matrices here: W , W , and W with shapes:
        - W : (NoOfHiddenNeurons, n )
        - W : (NoOfHiddenNeurons, NoOfHiddenNeurons)
        - W : (n , NoOfHiddenNeurons)
      - The weight matrix W is the memory the RNN is trying to maintain from the previous layers.
      - A lot of papers and books write the same architecture this way:
  - ![image](https://user-images.githubusercontent.com/36159918/206914489-8ee595ce-4129-4261-9647-2af5a74b321c.png)
    - It's harder to interpreter. It's easier to roll this drawings to the unrolled version.
  - In the discussed RNN architecture, the current output ?? depends on the previous inputs and activations.
  - Let's have this example 'He Said, "Teddy Roosevelt was a great president"'. In this example Teddy is a person name but
    we know that from the word president that came after Teddy not from He and said that were before it.
  - So limitation of the discussed architecture is that it can not learn from elements later in the sequence. To address this
    problem we will later discuss Bidirectional RNN (BRNN)
  - Now let's discuss the forward propagation equations on the discussed architecture:
  - ![image](https://user-images.githubusercontent.com/36159918/207071962-c2ae7150-3605-4aa4-a48b-0a79d4ce0a48.png)
  
  - The activation function of a is usually tanh or ReLU and for y depends on your task choosing some activation
    functions like sigmoid and softmax. In name entity recognition task we will use sigmoid because we only have two
    classes.
  - In order to help us develop complex RNN architectures, the last equations needs to be simplified a bit.
  - Simplified RNN notation:
    - ![image](https://user-images.githubusercontent.com/36159918/207072246-df50ff6e-7832-46f6-8a9b-34d3cb6653d4.png)
    
    - w is w and w stacked horizontally.
    - [a , x ] is a and x stacked vertically.
    - w shape: (NoOfHiddenNeurons, NoOfHiddenNeurons + n )
    - [a , x ] shape: (NoOfHiddenNeurons + n , 1)
 
  # Backpropagation through time
  - Let's see how backpropagation works with the RNN architecture.
  - Usually deep learning frameworks do backpropagation automatically for you. But it's useful to know how it works in
    RNNs.
  - Here is the graph:
    - ![image](https://user-images.githubusercontent.com/36159918/207072632-f69f54de-7ffb-4a66-9604-e991894cafb0.png)
    - Where w , b , w , and b are shared across each element in a sequence.
  - We will use the cross-entropy loss function:
    ![image](https://user-images.githubusercontent.com/36159918/207072767-0f87da35-50da-40e2-99d0-50bf0138a448.png)

    - Where the first equation is the loss for one example and the loss for the whole sequence is given by the summation
     over all the calculated single example losses.
  - Graph with losses:
    - ![image](https://user-images.githubusercontent.com/36159918/207072991-5dad62ca-b675-4f55-a450-b64092b9c801.png)

  - The backpropagation here is called backpropagation through time because we pass activation a from one sequence
    element to another like backwards in time.
  
  # Different types of RNNs
  
  - So far we have seen only one RNN architecture in which T equals T . In some other problems, they may not equal so we
need different architectures.
  - The ideas in this section was inspired by Andrej Karpathy blog. Mainly this image has all types:
    - ![image](https://user-images.githubusercontent.com/36159918/207073257-eca2153f-b781-49a7-be78-0070b60fe244.png)

  - The architecture we have descried before is called Many to Many.
  - In sentiment analysis problem, X is a text while Y is an integer that rangers from 1 to 5. The RNN architecture for that is
Many to One as in Andrej Karpathy image.
    - ![image](https://user-images.githubusercontent.com/36159918/207073416-2f049547-7e06-43eb-80df-9f755c58bcc3.png)

 - A One to Many architecture application would be music generation.
    - ![image](https://user-images.githubusercontent.com/36159918/207073495-43174ed7-d830-4683-a4ed-0278c6cf53b0.png)

      - Note that starting the second layer we are feeding the generated output back to the network.
- There are another interesting architecture in Many To Many. Applications like machine translation inputs and outputs
sequences have different lengths in most of the cases. So an alternative Many To Many architecture that fits the
translation would be as follows:
- ![image](https://user-images.githubusercontent.com/36159918/207073783-ea80cb8b-7073-49c7-9c90-445aec5378c7.png)

    - There are an encoder and a decoder parts in this architecture. The encoder encodes the input sequence into one
matrix and feed it to the decoder to generate the outputs. Encoder and decoder have different weight matrices.
 - Summary of RNN types:
  
 - ![image](https://user-images.githubusercontent.com/36159918/207073954-77ef6c7b-9c9c-4a16-9aa4-36913c98c23c.png)
  
  
  - There is another architecture which is the attention architecture which we will talk about in chapter 3.
 
 

# Language model and sequence generation
  - RNNs do very well in language model problems. In this section, we will build a language model using RNNs.
  - What is a language model
  
     - Let's say we are solving a speech recognition problem and someone says a sentence that can be interpreted into to
two sentences:
      - The apple and pair salad
      - The apple and pear salad
    
    - Pair and pear sounds exactly the same, so how would a speech recognition application choose from the two.
    - That's where the language model comes in. It gives a probability for the two sentences and the application decides
      the best based on this probability.
  - The job of a language model is to give a probability of any given sequence of words.
  - How to build language models with RNNs?
    - The first thing is to get a training set: a large corpus of target language text.
    - Then tokenize this training set by getting the vocabulary and then one-hot each word.
    - Put an end of sentence token <EOS> with the vocabulary and include it with each converted sentence. Also, use the
      token <UNK> for the unknown words.
  - Given the sentence "Cats average 15 hours of sleep a day. <EOS> "
    - In training time we will use this:
    - ![image](https://user-images.githubusercontent.com/36159918/207075283-12459a0d-7256-4476-84b7-15393f392909.png)

    - The loss function is defined by cross-entropy loss:
    - ![image](https://user-images.githubusercontent.com/36159918/207075334-535ac69f-8d6c-4247-9703-17763503f0eb.png)
       - i is for all elements in the corpus, t - for all timesteps.
  - To use this model:
    - For predicting the chance of next word, we feed the sentence to the RNN and then get the final y^<t> hot vector
and sort it by maximum probability.
    - For taking the probability of a sentence, we compute this:
      - p(y , y , y ) = p(y ) * p(y | y ) * p(y | y , y )
      - This is simply feeding the sentence into the RNN and multiplying the probabilities (outputs).
  
  # Sampling novel sequences
  
  - After a sequence model is trained on a language model, to check what the model has learned you can apply it to sample
    novel sequence.
  - Lets see the steps of how we can sample a novel sequence from a trained sequence language model:
    - Given this model:
    - ![image](https://user-images.githubusercontent.com/36159918/207075982-0e10e71f-d54f-4e33-acf5-cb7e7926b30e.png)

    - We first pass a = zeros vector, and x = zeros vector.
    - iii. Then we choose a prediction randomly from distribution obtained by ?? . For example it could be "The".
      - In numpy this can be implemented using: numpy.random.choice(...)
      - This is the line where you get a random beginning of the sentence each time you sample run a novel sequence.
    - We pass the last predicted word with the calculated a<1>
    - v. We keep doing 3 & 4 steps for a fixed length or until we get the <EOS> token.
    - vi. You can reject any <UNK> token if you mind finding it in your output.
 - So far we have to build a word-level language model. It's also possible to implement a character-level language model.
 - In the character-level language model, the vocabulary will contain [a-zA-Z0-9] , punctuation, special characters and
possibly token
  
 - Character-level language model has some pros and cons compared to the word-level language model
    - Pros:
        a. There will be no <UNK> token - it can create any word.
    - Cons:
      a. The main disadvantage is that you end up with much longer sequences.
      b. Character-level language models are not as good as word-level language models at capturing long range
        dependencies between how the the earlier parts of the sentence also affect the later part of the sentence.
      c. Also more computationally expensive and harder to train.
- The trend Andrew has seen in NLP is that for the most part, a word-level language model is still used, but as computers
get faster there are more and more applications where people are, at least in some special cases, starting to look at
more character-level models. Also, they are used in specialized applications where you might need to deal with
unknown words or other vocabulary words a lot. Or they are also used in more specialized applications where you have
a more specialized vocabulary.
  
  
  
# Vanishing gradients with RNNs
 
 - One of the problems with naive RNNs that they run into vanishing gradient problem.
 - An RNN that process a sequence data with the size of 10,000 time steps, has 10,000 deep layers which is very hard to
  optimize.
 - Let's take an example. Suppose we are working with language modeling problem and there are two sequences that
  model tries to learn:
  - "The cat, which already ate ..., was full"
  - "The cats, which already ate ..., were full"
  - Dots represent many words in between.
- What we need to learn here that "was" came with "cat" and that "were" came with "cats". The naive RNN is not very
  good at capturing very long-term dependencies like this.
- As we have discussed in Deep neural networks, deeper networks are getting into the vanishing gradient problem. That
  also happens with RNNs with a long sequence size.

  ![image](https://user-images.githubusercontent.com/36159918/207290787-9e2014bf-529c-4f4e-9971-ff770027a933.png)
  
- For computing the word "was", we need to compute the gradient for everything behind. Multiplying fractions tends to
vanish the gradient, while multiplication of large number tends to explode it.
  - Therefore some of your weights may not be updated properly.
- In the problem we descried it means that its hard for the network to memorize "was" word all over back to "cat". So in
  this case, the network won't identify the singular/plural words so that it gives it the right grammar form of verb
  was/were.
- The conclusion is that RNNs aren't good in long-term dependencies
- In theory, RNNs are absolutely capable of handling such ???long-term dependencies.??? A human could carefully pick
parameters for them to solve toy problems of this form. Sadly, in practice, RNNs don???t seem to be able to learn
them. http://colah.github.io/posts/2015-08-Understanding-LSTMs/
  
- Vanishing gradients problem tends to be the bigger problem with RNNs than the exploding gradients problem. We will
discuss how to solve it in next sections.
  
- Exploding gradients can be easily seen when your weight values become NaN . So one of the ways solve exploding
gradient is to apply gradient clipping means if your gradient is more than some threshold - re-scale some of your
gradient vector so that is not too big. So there are cliped according to some maximum value.
  
- ![image](https://user-images.githubusercontent.com/36159918/207294505-a32cf077-0588-4011-9bb7-208923626eac.png)

- Extra:
  - Solutions for the Exploding gradient problem:
    - Truncated backpropagation.
      - Not to update all the weights in the way back.
      - Not optimal. You won't update all the weights.
  - Gradient clipping.
  
  - Solution for the Vanishing gradient problem:
    - Weight initialization.
      - Like He initialization.
    - Echo state networks.
    - Use LSTM/GRU networks.
      - Most popular.
      - We will discuss it next.

  # Gated Recurrent Unit (GRU)
  
  - GRU is an RNN type that can help solve the vanishing gradient problem and can remember the long-term
    dependencies
  - The basic RNN unit can be visualized to be like this:
    ![image](https://user-images.githubusercontent.com/36159918/207298852-facfc7ad-6308-41f9-a6a7-2017ab6fbd7a.png)
  - We will represent the GRU with a similar drawings.
  - Each layer in GRUs has a new variable C which is the memory cell. It can tell to whether memorize something or not.
  - Equations of the GRUs:
   - ![image](https://user-images.githubusercontent.com/36159918/207299070-0dd0a3fe-e117-40f8-8127-781ca0b3c564.png)

    - The update gate is between 0 and 1
       - To understand GRUs imagine that the update gate is either 0 or 1 most of the time.
  -  So we update the memory cell based on the update cell and the previous cell.
  
  - Lets take the cat sentence example and apply it to understand this equations
     - Sentence: "The cat, which already ate ........................, was full"
     - We will suppose that U is 0 or 1 and is a bit that tells us if a singular word needs to be memorized
     - Splitting the words and get values of C and U at each place
  - ![Capture](https://user-images.githubusercontent.com/36159918/207300075-ad401a1d-ce24-4326-a80f-b79f12d7dd8f.PNG)
  -  Drawing for the GRUs
  - ![image](https://user-images.githubusercontent.com/36159918/207300201-59defaed-9c18-49dc-9464-cae30bbad6cb.png)

    - Drawings like in http://colah.github.io/posts/2015-08-Understanding-LSTMs/ is so popular and makes it easier to
      understand GRUs and LSTMs. But Andrew Ng finds it's better to look at the equations.
  
 - Because the update gate U is usually a small number like 0.00001, GRUs doesn't suffer the vanishing gradient problem.
  - In the equation this makes C = C<t-1> in a lot of cases.
- Shapes:
  - a shape is (NoOfHiddenNeurons, 1)
  - c is the same as a <t>
  - c ~<t> is the same as
  - u<t> is also the same dimensions of a <t>
  - The multiplication in the equations are element wise multiplication.
  - What has been descried so far is the Simplified GRU unit. Let's now describe the full one:
  - The full GRU contains a new gate that is used with to calculate the candidate C. The gate tells you how relevant is
    C <t-1> to C<t>
  - Equations:
  -  ![image](https://user-images.githubusercontent.com/36159918/207306497-0a445701-e24a-4fa1-a35f-e06e40e23337.png)

  - Shapes are the same
  - So why we use these architectures, why don't we change them, how we know they will work, why not add another gate,
    why not use the simpler GRU instead of the full GRU; well researchers has experimented over years all the various types
    of these architectures with many many different versions and also addressing the vanishing gradient problem. They have
    found that full GRUs are one of the best RNN architectures to be used for many different problems. You can make your
    design but put in mind that GRUs and LSTMs are standards.
  
  # Long Short Term Memory (LSTM)
  - LSTM - the other type of RNN that can enable you to account for long-term dependencies. It's more powerful and
    general than GRU.
  - In LSTM , C != a
  - Here are the equations of an LSTM unit:
   - ![image](https://user-images.githubusercontent.com/36159918/207307033-85b90cf1-4e85-4902-b1ee-54d327b542e7.png)
 
  - In GRU we have an update gate U , a relevance gate r , and a candidate cell variables C while in LSTM we have an
update gate U (sometimes it's called input gate I), a forget gate F , an output gate O , and a candidate cell variables C
  
  - Drawings (inspired by http://colah.github.io/posts/2015-08-Understanding-LSTMs/):
  - ![image](https://user-images.githubusercontent.com/36159918/207307189-bb402d06-5fc6-469b-8b69-f67992840dc6.png)

  - Some variants on LSTM includes:
    - LSTM with peephole connections.
      - The normal LSTM with C <t-1> included with every gate.
  - There isn't a universal superior between LSTM and it's variants. One of the advantages of GRU is that it's simpler and can
be used to build much bigger network but the LSTM is more powerful and general.
  
  
  # Bidirectional RNN
  - There are still some ideas to let you build much more powerful sequence models. One of them is bidirectional RNNs and
    another is Deep RNNs.
  - As we saw before, here is an example of the Name entity recognition task:
    - ![image](https://user-images.githubusercontent.com/36159918/207307608-345d92c5-bf9d-4fc3-b81e-4252a5207f82.png)

  -  The name Teddy cannot be learned from He and said, but can be learned from bears.
  - BiRNNs fixes this issue.
  - Here is BRNNs architecture:
  - ![image](https://user-images.githubusercontent.com/36159918/207307834-79884207-3b1c-40e7-89dc-e3ce36a9f02b.png)

  
  - Note, that BiRNN is an acyclic graph.
  - Part of the forward propagation goes from left to right, and part - from right to left. It learns from both sides
  - To make predictions we use ?? by using the two activations that come from left and right.
  - The blocks here can be any RNN block including the basic RNNs, LSTMs, or GRUs.
  - For a lot of NLP or text processing problems, a BiRNN with LSTM appears to be commonly used.
  - The disadvantage of BiRNNs that you need the entire sequence before you can process it. For example, in live speech
    recognition if you use BiRNNs you will need to wait for the person who speaks to stop to take the entire sequence and
    then make your predictions
  
  # Deep RNNs
  - In a lot of cases the standard one layer RNNs will solve your problem. But in some problems its useful to stack some
    RNN layers to make a deeper network.
  - For example, a deep RNN with 3 layers would look like this:
  
     ![image](https://user-images.githubusercontent.com/36159918/207308296-e0c51682-c334-448b-aa8c-b03a36f2b80c.png)

  - In feed-forward deep nets, there could be 100 or even 200 layers. In deep RNNs stacking 3 layers is already considered
      deep and expensive to train.
  - In some cases you might see some feed-forward network layers connected after recurrent cell.
  
  - Back propagation with RNNs
  - In modern deep learning frameworks, you only have to implement the forward pass, and the framework takes care
of the backward pass, so most deep learning engineers do not need to bother with the details of the backward
pass. If however you are an expert in calculus and want to see the details of backprop in RNNs, you can work
through this optional portion of the notebook.
  
  - The quote is taken from this notebook. If you want the details of the back propagation with programming notes look at
     the linked notebook.
  
   

  




