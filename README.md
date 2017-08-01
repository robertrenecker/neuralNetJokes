#Bingo Bango Bongo


"""
1) Tokenize text

2) Remove infrequent words

3) Prepend special START and END tokens. --> Learn/Tokenize the words that tend to start / end sentences.

4) Build the training data metrics --> learn the indices of most common / remembered words.

5) Input to model is going to be a sequence of words, a matrix X which will have a one-hot vectors
which will assign the word to the given indice. Otherwise "UNKNOWN_TOKEN" to the unknown_token word.

6) Initialize RNN Model, initialize instance variables, initialize parameters (U, V, W)
   --> Will implement a Theano version later...






"""
