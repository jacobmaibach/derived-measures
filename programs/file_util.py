import csv
import glob

def reverse_match(path_template):
    '''
    Takes a template with a signle wildcard and returns a list of matched labels,
    one for each file satisfying the template.
    '''
    field_count = path_template.count('*')
    assert(field_count == 1)
    parts = path_template.split('*')
    start_index = len(parts[0])
    end_index = -len(parts[1])
    listed = glob.glob(path_template)
    out = []
    for filename in listed:
        label = filename[start_index:end_index]
        out.append(label)
    return out

def load_documents(path_template,label_transform = None):
    '''
    Return a list of (sentence,label) pairs,
    where each label is the name of the document
    from which the sentence was taken.

    If the option label_transform is not None,
    it is applied to each document name to create the label.
    '''
    file_list = glob.glob(path_template)
    label_list = reverse_match(path_template)
    out = []
    for filename,label in zip(file_list,label_list):
        if(label_transform is not None):
            label = label_transform(label)
        with open(filename,'r') as file:
            lines = [l.strip() for l in file]
        sentences = [l for l in lines if l]
        out.extend([(s,label) for s in sentences])
    return out

###

def save_csv(pairs,path):
    with open(path,'w') as file:
        writer = csv.writer(file)
        writer.writerows(path)

def load_csv(path):
    with open(path,'r') as file:
        reader = csv.reader(file)
        return list(reader)

###

def denumber_transform(label,delim='_'):
    '''
    Standard label_transform to remove numbers/extraneous information from document labels.
    In particular, each document name is presumed to be of the form:
    label + delim + extra
    For example, "positive_1" is transformed to "positive".
    '''
    parts = label.split(delim)
    return parts[0]