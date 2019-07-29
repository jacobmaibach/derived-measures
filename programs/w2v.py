import gensim
from gensim.models import KeyedVectors

GN_PATH = '../models/external_w2v/GoogleNews-vectors-negative300.bin'

class WordModel:
    def __init__(self,model):
        self.model = model
        self.vocab = model.wv.vocab
        example_word = next(iter(self.vocab))
        self.zero = 0.0*self.model[example_word]

    def __getitem__(self,key):
        if(isinstance(key,str)):
            word = key
            return (self.model[word] if word in self.vocab else self.zero)
        else:
            word_list = key
            return [(self.model[word] if word in self.vocab else self.zero) for word in word_list]

def load_word_model():
    model = KeyedVectors.load_word2vec_format(GN_PATH, binary=True)
    return WordModel(model)

