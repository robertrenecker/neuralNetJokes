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
        raw_data = list(DictReader(open("rnn/data/example.csv", 'r')))
        self.reports = ["%s %s %s" % (START_SENT, x, END_SENT) for x in raw_data]


        #Tokenize sentences into words
        self.token_sentences = [nltk.word_tokenize(sent) for sent in self.reports]
        word_frequency = nltk.FreqDist(itertools.chain(*self.token_sentences))
        #print("Found %d unique words tokens." % len(word_frequency.items()))

        #Get most common words and build an index to word / word to index vectors

        vocab = word_frequency.most_common(vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])




if __name__ == "__main__":
    print("Working Fine")
