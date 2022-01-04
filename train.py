import torch
from torch.utils.data import DataLoader
from model import Net
from data import Data,vocab_size

train = Data(train=True)
test = Data(train=False)

trainloader = DataLoader(train,batch_size=128,shuffle=True)
testloader = DataLoader(train,batch_size=128,shuffle=True)

net = Net(n_inputs=vocab_size,hidden_size=128)

optimizer = torch.optim.Adam(net.parameters(),lr=5e-3)
lossfn = torch.nn.CrossEntropyLoss()

net.train_on(
    trainloader=trainloader,
    testloader=testloader,
    epochs=30,
    optimizer=optimizer,
    lossfn=lossfn,
)