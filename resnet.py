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

        - residual: boolean
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

    def forward(self, x, residual=False):
        z = self.fc1(x)
        z = self.bn1(z)
        z = self.relu1(z)

        z = self.fc2(z)
        z = self.bn2(z)

        # Shortcut connection
        # This if statement is the only difference between
        # a fully connected net and a resnet!
        if residual:
            z = self.shortcut(z, x)

        z = self.relu2(z)

        return z


class EndtoEndNet(nn.Module):
    def __init__(
            self,
            in_features, out_features, dims, n_blocks,
            residual=True
    ):
        """Architecture for learning end-to-end Behavorial Cloning
        Input -> ResNet -> Action

        set n_res_blocks=0 to get r3m_bc architecture
        """
        super().__init__()
        self.residual = residual

        # Input
        self.bn0 = nn.BatchNorm1d(in_features)
        self.fcIn = nn.Linear(in_features, 256, bias=False)
        self.bnIn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.stack = nn.ModuleList([block(256) for _ in range(n_blocks)])

        # Output
        self.fcOut = nn.Linear(256, out_features, bias=True)

    def forward(self, x):
        z = self.bn0(x)
        z = self.fcIn(z)
        z = self.bnIn(z)
        z = self.relu(z)

        for l in self.stack:
            z = l(z, residual=self.residual)

        z = self.fcOut(z)
        return z


class TransferableNet(nn.Module):
    def __init__(
            self,
            in_features, out_features, dims, n_blocks,
            residual=True
    ):
        """Architecture for learning transferable representation
        Learn representation without hand features to allow transferring to robot;
        Process hand features and the rest with 2 separate streams so that
        the non-hand feature extractor is transferable
        """
        super().__init__()
        r3m_dim, task_dim, hand_pose_dim, bbox_dim, cam_dim = dims
        self.hand_features = hand_pose_dim + bbox_dim
        self.r3m_features, self.task_features = r3m_dim, task_dim
        self.residual = residual

        # Non-hand input
        nonhand_features = in_features - self.hand_features
        self.nh_bn0 = nn.BatchNorm1d(nonhand_features)
        self.nh_fcIn = nn.Linear(nonhand_features, 256, bias=False)
        self.nh_bnIn = nn.BatchNorm1d(256)
        self.nh_relu = nn.ReLU()

        self.nh_stack = nn.ModuleList([block(256) for _ in range(n_blocks)])

        # Hand input
        self.h_bn0 = nn.BatchNorm1d(self.hand_features)
        self.h_fcIn = nn.Linear(self.hand_features, 256, bias=False)
        self.h_bnIn = nn.BatchNorm1d(256)
        self.h_relu = nn.ReLU()

        self.h_stack = nn.ModuleList([block(256) for _ in range(n_blocks)])

        # Output
        self.fcOut = nn.Linear(512, out_features, bias=True)

    def forward(self, x):
        # Get hand input
        hand_start = self.r3m_features + self.task_features
        hand_end = hand_start + self.hand_features
        hand_x = x[:, hand_start:hand_end]

        # Get non-hand input
        nonhand_inds = torch.ones(x.size(1), dtype=torch.bool)
        nonhand_inds[hand_start:hand_end] = False
        nonhand_x = x[:, nonhand_inds]

        # Hand forward
        h_z = self.h_bn0(hand_x)
        h_z = self.h_fcIn(h_z)
        h_z = self.h_bnIn(h_z)
        h_z = self.h_relu(h_z)

        for h_l in self.h_stack:
            h_z = h_l(h_z, residual=self.residual)

        # Non-hand forward
        nh_z = self.nh_bn0(nonhand_x)
        nh_z = self.nh_fcIn(nh_z)
        nh_z = self.nh_bnIn(nh_z)
        nh_z = self.nh_relu(nh_z)

        for nh_l in self.nh_stack:
            nh_z = nh_l(nh_z, residual=self.residual)

        z = torch.cat((nh_z, h_z), dim=1)
        z = self.fcOut(z)
        return z


if __name__ == '__main__':
    r3m_dim, task_dim, hand_pose_dim, bbox_dim, cam_dim = 2048, 4, 48, 4, 3
    input_dim = sum([r3m_dim, task_dim, hand_pose_dim, bbox_dim, cam_dim, cam_dim])
    output_dim = sum([hand_pose_dim, bbox_dim])
    resnet = EndtoEndNet(
        in_features=input_dim,
        out_features=output_dim,
        dims=(r3m_dim, task_dim, hand_pose_dim, bbox_dim, cam_dim),
        n_blocks=0,
        residual=False
    ).to('cuda')
    rand_tensor = torch.Tensor(4, input_dim).to('cuda')
    out_tensor = resnet(rand_tensor)
    print(f'out tensor size: {out_tensor.size()}')
