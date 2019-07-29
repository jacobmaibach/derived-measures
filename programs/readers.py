import csv,json
from pathlib import Path

#import jsonstreamer

MAIN_DIRECTORY = Path(__file__).absolute().parents[1]
DATA_DIRECTORY = MAIN_DIRECTORY / 'data'

def reader_combined():
    direc = DATA_DIRECTORY / 'combined'
    file_list = direc.glob('*_labelled.txt')
    lib = dict()
    out = {'sentences':[],'labels':[],'source':[]}
    for path in file_list:
        with open(path,'r') as file:
            reader = csv.reader(file,delimiter='\t')
            lib[path.stem] = list(reader)
    for key in lib:
        X,Y = zip(*lib[key])
        Y = [int(y) for y in Y]
        Z = [key]*len(Y)
        out['sentences'].extend(X)
        out['labels'].extend(Y)
        out['source'].extend(Z)
    return out

# def reader_yelp():
#     direc = DATA_DIRECTORY / 'yelp_dataset'
#     filename = 'review.json'
#     out = {'sentences':[],'labels':[]}
#     with open(direc/filename,'r') as file:
#         for line in file:
#             data = json.loads(line)
#             out['sentences'].append(data['text'])
#             out['labels'].append(data['stars'])
#     return out

def streamer_yelp():
    direc = DATA_DIRECTORY / 'yelp_dataset'
    filename = 'review.json'
    with open(direc/filename,'r') as file:
        for line in file:
            data = json.loads(line)
            yield (data['text'],round(data['stars']))

def reader_yelp():
    out = {'sentences':[],'labels':[]}
    for (sentence,label) in streamer_yelp():
        out['sentences'].append(sentence)
        out['labels'].append(label)
    return out

def streamer_imbd(train=True):
    direc = DATA_DIRECTORY / 'aclImdb'
    if(train):
        direc /= 'train'
    else:
        direc /= 'test'
    pos_dir = direc / 'pos'
    neg_dir = direc / 'neg'
    out = {'sentences':[],'labels':[]}
    for subdir in [pos_dir,neg_dir]:
        file_list = subdir.glob('*.txt')
        for path in file_list:
            name = path.stem
            score = int(name.split('_')[-1])
            with open(path,'r') as file:
                text = ''.join(file)
            yield (text,score)
