import random
import json

import torch
from torch.nn import function as F
from torch.autograd import Variable

###
_settings_path_ = '../settings/classifier_config.json'
with open(_settings_path_,'r') as file:
    settings = json.load(file)
###

class MLPClassifier(torch.nn.Module):
    learning_rate = 0.0001
    def __init__(self,input_dim,output_dim,hidden_layer_dim=(100,),epochs=10,encoder=None):
        super(MLPClassifier,self).__init__()
        dim_list = (input_dim,) + hidden_layer_dim
        self.lin = [torch.nn.Linear(dim_list[i],dim_list[i+1]) for i in range(len(dim_list)-1)]
        self.out = torch.nn.Linear(dim_list[-1],output_dim)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        self.epochs = epochs
        self.encoder = encoder

    def forward(self,X):
        if(self.encoder is not None):
            X = self.encoder(X)
        for layer in self.lin:
            X = F.relu(layer(X))
        return self.out(X)

    def encode(self,X,layer_number=-1):
        if(layer_number < 0):
            layer_number += len(self.lin)
        for i in range(layer_number):
            X = F.relu(self.lin[i](X))
        return self.lin[layer_number](X)

    def train(self,data,epochs=None):
        if(epochs is None):
            epochs = self.epochs
        X,y = data
        for k in range(epochs):
            pred = self.forward(X)
            self.optim.zero_grad()
            self.loss(pred,y).backward()
            self.optim.step()

    def train_subset(self,data,schedule):
        data_size = len(data[1])
        data_index = list(range(data_size))
        for wave in schedule:
            if(len(wave) == 2):
                prop,epochs = wave
                rep = 1
            elif(len(wave) == 3):
                prop,epochs,rep = wave
            sample_size = round(data_size*prop)
            for r in range(rep):
                sub_index = random.sample(data_index)
                sub = [X[i] for i in sub_index],[y[i] for i in sub_index]
                self.train(sub,epochs=epochs)

class Encoder:
    def __init__(self,model,layer_number=-1):
        self.model = model
        self.layer_number = layer_number

    def __call__(self,X):
        return self.model.encode(X,layer_number=self.layer_number)

def create_mlp(data):
    input_dim = data[0].shape[0]
    output_dim = len(set(data[1]))


