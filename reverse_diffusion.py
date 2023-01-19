#########################################
# Author:       Gentry Atkinson         #
# Date:         18 Jan, 2023            #
# Organization: Texas State University  #
#########################################

# Reverse diffusion learns to remove noise from a signal

import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001
MOMENTUM = 0.9

class Reverse_Diffuser(nn.Module):
    def __init__(self, X0:torch.Tensor) -> None:
        super().__init__()
        self.output_shape = (X0.shape)
        self.model = nn.Sequential(
            nn.LazyConv1d(out_channels=3, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.LazyLinear(out_features=6),
            nn.Sigmoid()
        )
        self.model = self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR, momentum=MOMENTUM)

    def forward(self, batch : torch.Tensor):
        batch = batch.to(device)
        return torch.reshape(self.model(batch).cpu(), batch.shape)

    def fit(self, batch : torch.Tensor, target : torch.Tensor):
        batch = batch.to(device)
        target = target.to(device)
        f = self.model(batch)
        self.optimizer.zero_grad()
        loss = self.criterion(f, target)
        loss.backward()
        self.optimizer.step()
        return loss

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
    
    f = Reverse_Diffuser(batch[0])
    new_batch = f(batch)
    print(new_batch)
    for _ in range(100):
        print('loss: ', f.fit(new_batch, batch))
        new_batch = f(batch)
    print(new_batch)
    print(batch)