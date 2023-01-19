#########################################
# Author:       Gentry Atkinson         #
# Date:         18 Jan, 2023            #
# Organization: Texas State University  #
#########################################

# Forward diffusion adds noise to a signal

import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Forward_Diffuser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.abs(torch.randn(1))

    def forward(self, batch : torch.Tensor):
        torch.manual_seed(torch.sum(batch))
        sd = torch.sqrt(torch.var(batch))
        return (batch + torch.randn(batch.shape)*sd*self.scale)



#Testing
if __name__ == '__main__':
    f = Forward_Diffuser()
    g = Forward_Diffuser()
    h = Forward_Diffuser()
    batch = torch.Tensor([
        [1, 2, 1, 2, 1, 2],
        [1, 3, 1, 3, 1, 2],
        [0, 2, 0, 2, 1, 1],
    ])
    batch = f(batch)
    print(batch)
    batch = g(batch)
    print(batch)
    batch = h(batch)
    print(batch)

    print('--------------')

    batch = torch.Tensor([
        [1, 2, 1, 2, 1, 2],
        [1, 3, 1, 3, 1, 2],
        [0, 2, 0, 2, 1, 1],
    ])
    batch = f(batch)
    print(batch)
    batch = g(batch)
    print(batch)
    batch = h(batch)
    print(batch)