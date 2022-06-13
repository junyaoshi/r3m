import torch
from torch import nn
from collections import OrderedDict

class block(nn.Module):
    def __init__(self, in_features, out_features):
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
        self.fc1 = nn.Linear(in_features, out_features, bias=False)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(out_features, out_features, bias=False)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.relu2 = nn.ReLU()

        # Shortcut downsampling
        self.downsample = None
        if in_features != out_features:
            self.downsample = nn.Sequential(OrderedDict([
                ('down_fc', nn.Linear(in_features, out_features, bias=False)),
                ('down_bn', nn.BatchNorm1d(out_features))
            ]))

    def shortcut(self, z, x):
        """
        Implements parameter free shortcut connection by identity mapping.
        If dimensions of input x are greater than activations then this
        is rectified by the projection shortcut in Eqn.(2)
        as described by option B in paper.

        Parameters:
        - x: tensor
             the input to the block
        - z: tensor
             activations of block prior to final non-linearity
        """
        if x.shape != z.shape:
            return z + self.downsample(x)
        else:
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


class FullyConnectedResNet8(nn.Module):
    def __init__(self, in_features, out_features, shortcuts=True):
        super().__init__()
        self.shortcuts = shortcuts
        n = 2

        # Input
        self.bn0 = nn.BatchNorm1d(in_features)
        self.fcIn = nn.Linear(in_features, 1024, bias=False)
        self.bnIn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

        # Stack1
        self.stack1a = block(1024, 512)
        self.stack1b = nn.ModuleList([block(512, 512) for _ in range(n - 1)])

        # Stack2
        self.stack2a = block(512, 256)
        self.stack2b = nn.ModuleList([block(256, 256) for _ in range(n - 1)])

        # Output
        self.fcOut = nn.Linear(256, out_features, bias=True)

    def forward(self, x):
        z = self.bn0(x)
        z = self.fcIn(z)
        z = self.bnIn(z)
        z = self.relu(z)

        z = self.stack1a(z, shortcuts=self.shortcuts)
        for l in self.stack1b:
            z = l(z, shortcuts=self.shortcuts)

        z = self.stack2a(z, shortcuts=self.shortcuts)
        for l in self.stack2b:
            z = l(z, shortcuts=self.shortcuts)

        z = self.fcOut(z)
        return z

class FullyConnectedResNet16(nn.Module):
    def __init__(self, in_features, out_features, shortcuts=True):
        super().__init__()
        self.shortcuts = shortcuts
        n = 2

        # Input
        self.bn0 = nn.BatchNorm1d(in_features)
        self.fcIn = nn.Linear(in_features, 1024, bias=False)
        self.bnIn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

        # Stack1
        self.stack1 = nn.ModuleList([block(1024, 1024) for _ in range(n)])

        # Stack2
        self.stack2a = block(1024, 512)
        self.stack2b = nn.ModuleList([block(512, 512) for _ in range(n - 1)])

        # Stack3
        self.stack3a = block(512, 256)
        self.stack3b = nn.ModuleList([block(256, 256) for _ in range(n - 1)])

        # Stack4
        self.stack4a = block(256, 128)
        self.stack4b = nn.ModuleList([block(128, 128) for _ in range(n - 1)])

        # Output
        self.fcOut = nn.Linear(128, out_features, bias=True)

    def forward(self, x):
        z = self.bn0(x)
        z = self.fcIn(z)
        z = self.bnIn(z)
        z = self.relu(z)

        for l in self.stack1:
            z = l(z, shortcuts=self.shortcuts)

        z = self.stack2a(z, shortcuts=self.shortcuts)
        for l in self.stack2b:
            z = l(z, shortcuts=self.shortcuts)

        z = self.stack3a(z, shortcuts=self.shortcuts)
        for l in self.stack3b:
            z = l(z, shortcuts=self.shortcuts)

        z = self.stack4a(z, shortcuts=self.shortcuts)
        for l in self.stack4b:
            z = l(z, shortcuts=self.shortcuts)

        z = self.fcOut(z)
        return z

if __name__ == '__main__':
    in_features, out_features = 2100, 52
    resnet8 = FullyConnectedResNet8(
        in_features=in_features, out_features=out_features
    ).to('cuda')
    rand_tensor = torch.Tensor(4, 2100).to('cuda')
    out_tensor = resnet8(rand_tensor)
    print(f'out tensor size: {out_tensor.size()}')
