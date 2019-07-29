from file_util import load_csv,save_csv,load_documents

from s2v import load_sentence_model

label_map = dict()
inverse_label_map = dict()

def load_text_data(path,form='csv'):
    base = load_csv(path)
    X = sentence_model.embed(base[0])
    y = convert_labels(base[1])
    return X,y

def convert_labels(listed):
    out = []
    for name in listed:
        if(name not in label_map):
            val = 0 if not label_map else max(label_map.values())+1
            label_map[name] = val
        else:
            val = label_map[name]
        out.append(val)
    return out

def create_ordinal_encoding(values,vec_type=list):
    dim = len(values)-1
    code = dict()
    for i,val in values:
        vec = [0]*dim
        for j in range(i):
            vec[j] = 1
        vec = vec_type(vec)
        code[val] = vec
    return code

###

sentence_model = load_sentence_model()