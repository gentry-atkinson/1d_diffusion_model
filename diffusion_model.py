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
from sklearn.preprocessing import MinMaxScaler

NUM_LAYERS = 3

class Diffuser(nn.Module):
    def __init__(self, X0 : torch.Tensor) -> None:
        super().__init__()
        self.forward_layers = [Forward_Diffuser() for _ in range(NUM_LAYERS)]
        self.reverse_layers = [Reverse_Diffuser(X0) for _ in range(NUM_LAYERS)]

    def forward(self, X):
        encoded = X
        for f in self.forward_layers:
            encoded = f(encoded)
        decoded = encoded
        for r in self.reverse_layers[::-1]:
            decoded = r(decoded)
        return encoded, decoded

    def fit(self, X : torch.Tensor):
        pass

#Testing
if __name__ == '__main__':
    batch = [
        [1, 2, 1, 2, 1, 2],
        [1, 3, 1, 3, 1, 2],
        [0, 2, 0, 2, 1, 1],
    ]
    scaler = MinMaxScaler(feature_range=(0,1))
    batch = scaler.fit_transform(batch)
    batch = torch.Tensor(batch)

    d = Diffuser(batch[0])
    print(d(batch))
