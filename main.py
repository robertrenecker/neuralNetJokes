#NeuralNet
"""
csv.reader(csvfile, dialect='excel', **fmtparams) --> return a reader object which will iterate over lines in the given csv file...
csvfile can be any object which support the iterator protocol... returns a string each time its next() method is called.

"""
from nltk import sent_tokenize, word_tokenize
import nltk
import csv
from csv import DictReader
import re, itertools, sys
from sklearn.externals import joblib
import numpy as np
START_SENT = "SENT_START"
END_SENT = "SENT_END"
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class report:
    def __init__(self):
        raw_data = DictReader(open("rnn/data/reddit-comments-2015-08.csv", 'r'))
        data = []
        #BUILD/EXTEND SENTENCES WITHOUT START / END TOKEN
        for x in raw_data:
            data.extend(nltk.sent_tokenize(x['body']))
        #PUT START TOKEN AND END TOKEN ON EACH SENTENCE
        self.token_sent = ["%s %s %s" % (START_SENT, x, END_SENT) for x in data]


        #Tokenize sentences into words
        self.token_word = [nltk.word_tokenize(sent) for sent in self.token_sent]
        word_frequency = nltk.FreqDist(itertools.chain(*self.token_word))
        #print("Found %d unique words tokens." % len(word_frequency.items()))

        #Get most common words and build an index to word / word to index vectors

        vocab = word_frequency.most_common(vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)

        #DICT = [(WORD, COUNT) FOR ENUMERATES RETURN OF (COUNT, WORD)]
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        for i, sent in enumerate(self.token_word):
            self.token_word[i] = [w if w in word_to_index else unknown_token for w in sent]

        self.X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in self.token_word])
        print(self.X_train[10])
        self.y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in self.token_word])



class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        #Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        #randomly initialzie the paramteres...

        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        #T = The total number of time steps
        T = len(x)
        print(T)
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        o = np.zeros((T, self.word_dim))

        #FOR EACH TIME STEP
        for t in np.arange(T):
            #indexing U by x[t]. This is the same as multiplying U with a one-hot vector...
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o,s]

    def predict(self, x):
        #Perform forward propagation & return index of highest score.
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_loss(self, x, y):
        #initialize Loss
        L = 0
        #For each sentence...
        #print(len(y))
        for i in np.arange(len(y)):
            (o, s) = self.forward_propagation(x[i])
            #We only care about the 'correct predctions'
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            #Add to the loss based on how off we were...
            L += -1 * np.sum(np.log(correct_word_predictions))
        N = np.sum((len(yi) for yi in y))
        return L/N


    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]


    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        bptt_gradients = self.bptt(x, y)
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            parameter = operator.attrgetter(pname)(self)
            #print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                original_value = parameter[ix]
                # estmate: (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                parameter[ix] = original_value
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    #print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    #print "+h Loss: %f" % gradplus
                    #print "-h Loss: %f" % gradminus
                    #print "Estimated_gradient: %f" % estimated_gradient
                    #print "Backpropagation gradient: %f" % backprop_gradient
                    #print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            #print "Gradient check for parameter %s passed." % (pname)

    def sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= (learning_rate * dLdU)
        self.V -= (learning_rate * dLdV)
        self.W -= (learning_rate * dLdW)

    def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        #TODO: Add momentum!
        losses = []
        ex_seen = 0
        for epoch in range(nepoch):
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((ex_seen, loss))
                #print "Loss after num examples seen = %d epoch = %d: %f" % (ex_seen, epoch, loss)
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    #print "Setting learning rate to: ", learning_rate
                sys.stdout.flush()
            for i in range(len(y_train)):
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                ex_seen += 1
    def generate_sentence(t, model):
        new_sentence = [t.word_to_index[START]]
        while not new_sentence[-1] == t.word_to_index[END]:
            next_word_probs = model.forward_prop(new_sentence)
        #print next_word_probs[0][-1]
            sampled_word = t.word_to_index[unknown_token]
            while sampled_word == t.word_to_index[unknown_token]:
                samples = np.random.multinomial(5, next_word_probs[0][-1])
                sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [t.index_to_word[x] for x in new_sentence[1:-1]]
        return sentence_str



if __name__ == "__main__":
    egg = report()
    print("Report is done gathering training data")
    np.random.seed(10)
    model = RNNNumpy(vocabulary_size)
    print("Model Working")

    losses = model.train_with_sgd(egg.X_train[:5000], egg.y_train[:5000], nepoch=40, evaluate_loss_after=1)
    joblib.dump(model, 'trained_model_update.pkl')
    """m = joblib.load('trained_model_update.pkl')"""

    """
    for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) &lt; senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)

    """

    """
    st = ""
    for s in generate_sentence(egg, m):
        if (s in ['!', '#', ':', '@', '.', '\'']):
            st += s
        else:
            st += s + " "
    print(st)
    """
