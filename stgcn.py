import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging





torch.cuda.is_available()
class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        a =(0,2)
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),padding=a)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),padding=a)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),padding=a)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        # print('A_hat:')
        # print(A_hat.size())
        # print('t.')
        # print(t.size())
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # print('A_hat_after:')
        # print(A_hat.size())
        # print('t.permute')
        # print(t.permute(1, 0, 2, 3).size())
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block3 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)

        self.block4 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input +16) * 64,
                               num_timesteps_output)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        # out1.size()
        # print(A_hat.size())
        out2 = self.block2(out1, A_hat)
        out3 = self.block3(out2, A_hat)
        out4 = self.block3(out3, A_hat)
        out5 = self.last_temporal(out4)
        out6 = self.fully(out4.reshape((out5.shape[0], out5.shape[1], -1)))

        return out6
# class STGCN(nn.Module):
#     """
#     Spatio-temporal graph convolutional network as described in
#     https://arxiv.org/abs/1709.04875v3 by Yu et al.
#     Input should have shape (batch_size, num_nodes, num_input_time_steps,
#     num_features).
#     """
#
#     def __init__(self, num_nodes, num_features, num_timesteps_input,
#                  num_timesteps_output):
#         """
#         :param num_nodes: Number of nodes in the graph.
#         :param num_features: Number of features at each node in each time step.
#         :param num_timesteps_input: Number of past time steps fed into the
#         network.
#         :param num_timesteps_output: Desired number of future time steps
#         output by the network.
#         """
#         super(STGCN, self).__init__()
#         self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
#                                  spatial_channels=16, num_nodes=num_nodes)
#         self.block2 = STGCNBlock(in_channels=64, out_channels=64,
#                                  spatial_channels=16, num_nodes=num_nodes)
#         self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
#         self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
#                                num_timesteps_output)
#
#     def forward(self, A_hat, X):
#         """
#         :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
#         num_features=in_channels).
#         :param A_hat: Normalized adjacency matrix.
#         """
#         out1 = self.block1(X, A_hat)
#         print(out1.size())
#         out2 = self.block2(out1, A_hat)
#         print(out2.size())
#         out3 = self.last_temporal(out2)
#         print(out3.size())
#         out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
#         print(out4.size())
#         return out4


if __name__ == "__main__":
 # test here
     batch_size = 5
     n_vertex =4
     n_step = 12
     n_output = 3
     n_channel =2
     test_input = torch.randn(batch_size, n_vertex, n_step, n_channel)
     test_adj = torch.randn(n_vertex, n_vertex)
     test_network = STGCN(n_vertex, n_channel, n_step, n_output)
     test_output = test_network( test_adj ,test_input)


