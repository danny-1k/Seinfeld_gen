import torch
from model import Net
import data

net = Net(n_inputs=data.vocab_size,hidden_size=128)
net.load_model()

start = 'the meaning of life is'

hidden = None

out = start

with torch.no_grad():
    for char in start:
        x = data.hot_encode(char).unsqueeze(0)
        _,hidden = net(x,hidden)

    x = data.hot_encode(char[-1]).unsqueeze(0)


    for i in range(1000):
        pred,hidden = net(x,hidden)
        pred = pred.exp()
        pred_idx = torch.multinomial(pred, 1)[0]

        pred_char = data.vocab[pred_idx]

        out+= pred_char
        x=data.hot_encode(pred_char).unsqueeze(0)


print(out)