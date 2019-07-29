import functools

from w2v import load_word_model

class SentenceModel:
    def __init__(self,weight_func,word_model):
        self.word_model = word_model
        self.weight = weight_func

    def preprocess(self,sentence_list):
        punc = '.,;:!?\"\'()[]'
        out = []
        for sentence in sentence_list:
            for p in punc:
                sentence.replace(p,'')
            token_list = sentence.split()
            out.append(token_list)
        return out

    def vectorize(self,tokens):
        out = []
        for token_list in tokens:
            weight_list = self.weight(token_list)
            vec_list = [a*v for (a,v) in zip(weight_list,self.word_model[token_list])]
            total = functools.reduce(lambda u,v:u+v,vec_list)
            out.append(total)
        return out

    def embed(self,sentence_list):
        tokens = self.preprocess(sentence_list)
        return self.vectorize(tokens)

def avg_weight(listed):
    n = len(listed)
    return [1/n for w in listed]

def load_sentence_model():
    word_model = load_word_model()
    return SentenceModel(weight_func=avg_weight,word_model=word_model)
