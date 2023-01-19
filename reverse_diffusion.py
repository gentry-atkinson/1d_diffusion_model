#########################################
# Author:       Gentry Atkinson         #
# Date:         18 Jan, 2023            #
# Organization: Texas State University  #
#########################################

# Reverse diffusion learns to remove noise from a signal

import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Reverse_Diffuser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(3, 3, 3),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )
        self.model = self.model.to(device)

    def forward(self, batch : torch.Tensor):
        batch = batch.to(device)
        return self.model(batch).cpu()

    def fit(self, batch : torch.Tensor, target : torch.Tensor):
        pass

#Testing
if __name__ == '__main__':
    f = Reverse_Diffuser()
    batch = torch.Tensor([
        [1, 2, 1, 2, 1, 2],
        [1, 3, 1, 3, 1, 2],
        [0, 2, 0, 2, 1, 1],
    ])
    batch = f(batch)
    print(batch)