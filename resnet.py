import torch
from torch import nn

class block(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        """
        A 2-layer residual learning building block as illustrated by Fig.2
        in "Deep Residual Learning for Image Recognition"

        Parameters:

        - in_features:  int
                        the number of input features of this block
        - out_features: int
                        the number of output features of this block

        Attributes:

        - shortcuts: boolean
                     When false the residual shortcut is removed
                     resulting in a 'plain' block.
        """
        # Setup layers
        self.fc1 = nn.Linear(num_features, num_features, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_features, num_features, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features)
        self.relu2 = nn.ReLU()

    def shortcut(self, z, x):
        """
        Implements parameter free shortcut connection by identity mapping.

        Parameters:
        - x: tensor
             the input to the block
        - z: tensor
             activations of block prior to final non-linearity
        """
        return z + x

    def forward(self, x, shortcuts=False):
        z = self.fc1(x)
        z = self.bn1(z)
        z = self.relu1(z)

        z = self.fc2(z)
        z = self.bn2(z)

        # Shortcut connection
        # This if statement is the only difference between
        # a fully connected net and a resnet!
        if shortcuts:
            z = self.shortcut(z, x)

        z = self.relu2(z)

        return z


class FullyConnectedResNet(nn.Module):
    def __init__(self, in_features, out_features, n_res_blocsk, shortcuts=True):
        super().__init__()
        self.shortcuts = shortcuts

        # Input
        self.bn0 = nn.BatchNorm1d(in_features)
        self.fcIn = nn.Linear(in_features, 256, bias=False)
        self.bnIn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.stack = nn.ModuleList([block(256) for _ in range(n_res_blocsk)])

        # Output
        self.fcOut = nn.Linear(256, out_features, bias=True)

    def forward(self, x):
        z = self.bn0(x)
        z = self.fcIn(z)
        z = self.bnIn(z)
        z = self.relu(z)

        for l in self.stack:
            z = l(z, shortcuts=self.shortcuts)

        z = self.fcOut(z)
        return z

if __name__ == '__main__':
    in_features, out_features = 2100, 52
    resnet = FullyConnectedResNet(
        in_features=in_features, out_features=out_features, n_res_blocsk=8
    ).to('cuda')
    rand_tensor = torch.Tensor(4, 2100).to('cuda')
    out_tensor = resnet(rand_tensor)
    print(f'out tensor size: {out_tensor.size()}')
