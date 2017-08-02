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

   --> For activation function TanH(U*xt + W*st-1), it is best to initialize our parameter weights from
   [-1/sqrt(n) , 1/sqrt(n)]... where n is the number of incoming connections from the previous
   layer.

   hidden_dim =>  the size of our hidden layer that we choose
   word_dim => the size of our vocabulary


   7) Forward Propagation

      --> Predicting word probabilities...

      --> We save all hidden states in s because we need them later. Initial hidden layer
      initialized = 0

  8) Prediction



  9) Loss function

    --> Calculate the loss, measure errors the model makes.

    --> Find the parameters U, V and W that minimize the loss function for our training data.

    --> (Cross Entropy Loss): if we have N training examples (words in our text) and C classes (the size of our vocabulary),
    then the loss with respect to our predictions o and the true labels y is given by:
          --> L(y,o) = (- 1/N) * Sum(y'n*log(o'n))

          Implement with calculate_total_loss


  10) Back Propagation Through Time (BPTT)
   


































"""
