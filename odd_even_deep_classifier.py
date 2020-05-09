import sys

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
EPOCHS = 20
THENUMBER = 0

class NumberDataset(Dataset):
    def __init__(self, at_most_size, seed=0):
        # Seed for reproducibility
        np.random.seed(seed)
        # Why this range? Idk, will not mention this in the paper 
        self.dataset = np.random.randint(-2**32 ,
                                         2**32 -1,
                                         size=at_most_size,
                                         dtype=np.float32)
        self.dataset = np.delete(self.dataset,
                                 np.where(self.dataset == 0))
        self.answers = ((self.dataset % 2) == 0).astype(np.float32)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return (self.dataset[idx], self.answers[idx])

class OEDCC(nn.Module):
    """
    Odd-Even Deep Cool Classifier Class
    """
    def __init__(self, layer_size, no_layers):
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
                                    nn.Softmax(dim=-1))

    def forward(self, number):
        if use_cuda:
           mlp_input = Variable(number).cuda() 
        else:
           mlp_input = Variable(number)

        predictions = self.mlp(mlp_input)
        return predictions
        
if __name__ == '__main__':

    dataset = NumberDataset(10e7)
    
    model = OEDCC(129, 51)

    if USE_CUDA:
        model.cuda()
        model = DataParallel(model)

    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), 1e-4)

    for epoch in range(EPOCHS):

        if USE_CUDA:
            train_loss = torch.cuda.FloatTensor()
        else:
            train_loss = torch.FloatTensor()

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0 if sys.gettrace() else 4,
            pin_memory=USE_CUDA)

        for i_batch, sample in enumerate(dataloader):       
            nums = sample[0]
            ans = sample[1]

            pred = model(Variable(sample))
            loss = loss_function(pred, ans)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = torch.cat([train_loss, loss.data])
            print("Epoch {}, Train Loss: {}".format(epoch, torch.mean(train_loss)))


    #model.eval()
    #is_odd( 
