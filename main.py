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
example_text = "Hello Mr. Smith, how are you today? I am feeling quite well, cheers! What appeared last night?"
START_SENT = "SENT_START"
END_SENT = "SENT_END"
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"


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

        for i, sent in enumerate(self.token_sent):
            self.token_sent[i] = [w if w in word_to_index else unknown_token for w in sent]

        self.X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in self.token_sent])
        self.y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in self.token_sent])



class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        #Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dem
        self.bptt_truncate = bptt_truncate

        #randomly initialzie the paramteres...

        




if __name__ == "__main__":
    print("Working Fine")
    egg = report()
