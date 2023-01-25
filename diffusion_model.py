#########################################
# Author:       Gentry Atkinson         #
# Date:         19 Jan, 2023            #
# Organization: Texas State University  #
#########################################

# Combine the forward and reverse proceses

from forward_diffusion import Forward_Diffuser
from reverse_diffusion import Reverse_Diffuser
import torch
from torch import nn
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt

NUM_LAYERS = 3
EPOCHS = 10

class Diffuser(nn.Module):
    def __init__(self, X0 : torch.Tensor) -> None:
        super().__init__()
        self.forward_layers = Forward_Diffuser()
        self.reverse_layers = Reverse_Diffuser(X0)

    def forward(self, X):
        encoded = X
        for _ in range(NUM_LAYERS):
            encoded = self.forward_layers(encoded)
        decoded = encoded
        for _ in range(NUM_LAYERS):
            decoded = self.reverse_layers(decoded)
        return encoded, decoded

    def fit(self, X : torch.Tensor):
        #set up dataloader

        #for e in epochs
        for _ in range(EPOCHS):
            #for batch in X
            for i in range(NUM_LAYERS):
                encoded = self.forward_layers(X)
                self.reverse_layers.fit(encoded, X)
                X = encoded

#Testing
if __name__ == '__main__':
    batch = [
        [1, 2, 1, 2, 1, 2],
        [1, 3, 1, 3, 1, 2],
        [0, 2, 0, 0, 1, 1],
    ]
    batch = minmax_scale(batch, feature_range=(0,1), axis=-1)
    batch = torch.Tensor(batch)
    print(batch)

    f = Forward_Diffuser()
    enc = batch
    plt.figure()
    color = ['green', 'blue', 'red']
    for _ in range(3):  
        enc = f(enc)
        for i, sig in enumerate(enc):
            print(sig.shape)
            plt.plot(range(len(sig)), sig, c=color[i])
    plt.show()

    d = Diffuser(batch[0])
    print(d(batch))
    d.fit(batch)
    print(d(batch))
    
