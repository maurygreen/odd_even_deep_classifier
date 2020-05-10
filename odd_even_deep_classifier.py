import sys
import os

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.optim as optim

import numpy as np
import json

from time import time

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
EPOCHS = 10
THENUMBER = 0

def accuracy(pred, ans):
    pass

class NumberDataset(Dataset):
    def __init__(self, at_most_size, seed=0):
        # Seed for reproducibility
        np.random.seed(seed)
        # Why this range? Idk, will not mention this in the paper 
        self.dataset = np.random.randint(-2**32 ,
                                         2**32 -1,
                                         size=at_most_size)
        remove = np.where(self.dataset == 0)
        self.answers = ((self.dataset % 2) == 0).astype(np.float32)
        self.dataset = np.delete(self.dataset, remove)
        self.dataset = self.dataset.astype(np.float32)
        self.answers = np.delete(self.answers, remove)

        #fname = 'num_data.json'
        #if os.path.exists('num_data.json'):
        #    with open(fname, 'r') as fl:
        #        data = json.load(fl)
        #    self.dataset = data['nums']
        #    self.answers = data['ans']
        #else:
        #    data = {'nums': self.dataset.tolist(), 'ans':self.answers.tolist()}
        #    with open(fname, 'w') as fl:
        #        json.dump(data, fl)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        data = np.reshape(self.dataset[idx], (-1, 1))
        ans = np.reshape(self.answers[idx], (-1, 1))
        return (data, ans)

class OEDCC(nn.Module):
    """
    Odd-Even Deep Cool Classifier Class
    """
    def __init__(self, layer_size, no_layers):
        super(OEDCC, self).__init__()
        self.mlp = nn.Sequential()

        layer_id = 0
        for i in range(no_layers):
            if i == 0:
                self.mlp.add_module(str(layer_id),
                                    nn.Linear(1, layer_size))
            elif i == no_layers - 1:
                self.mlp.add_module(str(layer_id),
                                    nn.Linear(layer_size, 1))
            else:
                self.mlp.add_module(str(layer_id),
                                    nn.Linear(layer_size, layer_size))

            layer_id += 1
            if i < no_layers - 1:
                self.mlp.add_module(str(layer_id),
                                    nn.ReLU())
                layer_id += 1
            else:
                self.mlp.add_module(str(layer_id),
                                    nn.Sigmoid())

    def forward(self, number):
        if USE_CUDA:
           mlp_input = Variable(number).cuda() 
        else:
           mlp_input = Variable(number)

        predictions = self.mlp(mlp_input)
        return predictions

def gimme():
    dataset = NumberDataset(int(10e4))

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False)
    return dataloader
        
if __name__ == '__main__':

    dataset = NumberDataset(int(10e5))
    
    model = OEDCC(48, 51)

    if USE_CUDA:
        model.cuda()
        model = DataParallel(model)
        print(model)

    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), 1e-4)

    for epoch in range(EPOCHS):
        if USE_CUDA:
            train_loss = torch.cuda.FloatTensor()
        else:
            train_loss = torch.FloatTensor()

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=512,
            shuffle=True,
            num_workers=0 if sys.gettrace() else 4,
            pin_memory=USE_CUDA)

        start = time()
        for i_batch, sample in enumerate(dataloader):       
            nums = sample[0]
            ans = sample[1]

            pred = model(Variable(nums))
            loss = loss_function(pred, Variable(ans).cuda() if USE_CUDA else Variable(ans)).unsqueeze(0)
            #loss = loss_function(pred.argmax(dim=-1), Variable(ans).cuda() if USE_CUDA else Variable(ans)).unsqueeze(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = torch.cat([train_loss, loss.data])
        elapsed = time() - start
        print("Epoch {}, Train Loss: {}, Elapsed time {}".format(epoch, torch.mean(train_loss), elapsed))

    torch.save(model.state_dict(), 'oedcc')
    #model.eval()
    #is_odd( 
