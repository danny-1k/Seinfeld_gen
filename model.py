import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from tqdm import tqdm

class Net(nn.Module):
    def __init__(self,n_inputs,hidden_size,dropout_p=0,n_layers=1):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.n_layers = n_layers

        self.rnn = nn.LSTM(n_inputs,hidden_size,n_layers,dropout=dropout_p,batch_first=True)
        self.fc = nn.Linear(hidden_size,n_inputs)


    def forward(self,x,hidden=None):
        out,hidden = self.rnn(x,hidden)
        
        out = out.contiguous().view(-1,self.hidden_size)

        out = self.fc(out)

        return out,hidden

    def save_model(self):
        torch.save(self.state_dict(),'model.pt')

    def load_model(self):
        self.load_state_dict(torch.load('model.pt',map_location=torch.device('cpu')))

    def train_on(self,trainloader,testloader,epochs,optimizer,lossfn):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.to(device)
        
        train_loss_over_time = []
        test_loss_over_time = []

        for epoch in tqdm(range(epochs)):
            
            batch_train_loss = []
            batch_test_loss = []

            self.train()
            for x,y in trainloader:

                x = x.to(device).float()
                y = y.to(device).long()

                pred,_ = self.__call__(x)

                #y of shape (batch_size,seq_len)

                loss = lossfn(pred,y.view(-1))

                loss.backward()

                optimizer.step()

                optimizer.zero_grad()

                batch_train_loss.append(loss.item())
            
            self.eval()
            with torch.no_grad():
                for x,y in testloader:

                    x = x.to(device).float()
                    y = y.to(device).long()

                    pred,_ = self.__call__(x)

                    loss = lossfn(pred,y.view(-1))

                    batch_test_loss.append(loss.item())

            
            train_loss_over_time.append(sum(batch_train_loss)/len(batch_train_loss))
            test_loss_over_time.append(sum(batch_test_loss)/len(batch_test_loss))


            plt.plot(train_loss_over_time,color='red',label='Train loss')
            plt.plot(test_loss_over_time,color='blue',label='Test loss')

            plt.legend()

            plt.savefig('loss.png')

            plt.close('all')

            if len(test_loss_over_time) == 1 or (test_loss_over_time[-1] < test_loss_over_time[-2]):
                self.save_model()