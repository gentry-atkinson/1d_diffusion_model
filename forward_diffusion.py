#########################################
# Author:       Gentry Atkinson         #
# Date:         18 Jan, 2023            #
# Organization: Texas State University  #
#########################################

# Forward diffusion adds noise to a signal

from torch import nn

class forward_diffuser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
