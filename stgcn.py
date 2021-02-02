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
        a =(0,1)
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
        # self.LSTMBlock=LSTMBlock()
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

class LSTMBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_layer, out_channels):
        super(LSTMBlock, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_channels, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(in_channels, out_channels)
        # 默认输入数据格式:
        # input(seq_len, batch_size, input_size)
        # h0(num_layers * num_directions, batch_size, hidden_size)
        # c0(num_layers * num_directions, batch_size, hidden_size)
        # 默认输出数据格式：
        # output(seq_len, batch_size, hidden_size * num_directions)
        # hn(num_layers * num_directions, batch_size, hidden_size)
        # cn(num_layers * num_directions, batch_size, hidden_size)

        # batch_first=True 在此条件下，batch_size是处在第一个维度的。
        """
             # :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
             # num_features=in_channels)
             # :return: Output data of shape (batch_size, num_nodes,
             # num_timesteps_out, num_features_out=out_channels)
             # """

    def forward(self, x):
        # print(x.size())
        # x.size()
        ## Batchsize x N-Nodes x TimeStep x Channel => TimeStep x Batchsize x N-Nodes x Channel
        x = x.permute(2,0,1,3)
        # print(x.size())
        # x = x.reshape([x.shape[0], x.shape[1] ,x.shape[3] * x.shape[2]])
        # x = x.reshape([x.shape[0], x.shape[1] * x.shape[2], x.shape[3]])
        t, b, n, c =  x.shape
        x = x.reshape(t, b*n, c)
        # x = x.reshape([x.shape[0], x.shape[1] , x.shape[3]])
        # print(x.size())
        # x = x.reshape([x.shape[0], -1, x.shape[3]])
        # x = x.resize(,x.shape[2],x.shape[3])
        # x = x.permute(1,0,2)
        out, (h_n, c_n) = self.lstm(x)
        ## TimeStep x Batchsize x N-Nodes x Channel => Batchsize x N-Nodes x TimeStep x Channel
        out = out.reshape(t, b, n, c)
        out = out.permute(1, 2, 0, 3)
        # print(out.size())
            # 此时可以从out中获得最终输出的状态h
            # x = out[:, -1, :]
        # x = h_n[-1, :, :]
        # x = self.classifier(x)

        return out



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
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.LSTMBlock = LSTMBlock(in_channels=64, hidden_dim=64 , n_layer=3 ,out_channels= 64)
        self.fully = nn.Linear((num_timesteps_input) * 64,
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
        # print(out3.size())


        # out4 = self.last_temporal(out3)
        # print(out4.size())
        # out5 = self.fully(out4.reshape((out4.shape[0], out4.shape[1], -1)))
        # print(out5.size())
        # return out5
        out4 = self.last_temporal(out3)
        # print(out4.size())
        out5 = self.LSTMBlock(out4)
        # print(out5.size())
        # out6 = self.fully(out5.reshape((out5.shape[1], 4,out5.shape[0],64)))

        out6 = self.fully(out5.reshape((out5.shape[0], out5.shape[1], -1)))
        # print(out6.size())
        # out6 = self.fully(out5)
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


