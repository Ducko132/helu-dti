import torch
from torch_geometric.nn import to_hetero, SAGEConv
from torch.nn import Linear, Dropout


class GNNEncoder(torch.nn.Module):
  """
  Implements the Heterogeneous Graph Neural Network
  """

  def __init__(self, hidden_channels, out_channels, p=0.2):
    super().__init__()
    self.conv1 = SAGEConv((-1, -1), 2*hidden_channels, aggr='sum')
    self.conv2 = SAGEConv((-1, -1), out_channels, aggr='sum')
    self.dropout = Dropout(p)

  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index).relu()
    x = self.dropout(x)
    x = self.conv2(x, edge_index)
    return x
