import numpy as np
import torch
from torch_geometric.nn import to_hetero, SAGEConv
from torch.nn import Linear, Dropout
import random
from .edge_classifier import EdgeClassifier
from .gnn_encoder import GNNEncoder


class Model(torch.nn.Module):

  """
  Implements the HeLU-DTI model
  """

  def __init__(self, hidden_channels, data):
    super().__init__()
    self.encoder = GNNEncoder(hidden_channels, hidden_channels)

    self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
    self.decoder = EdgeClassifier(hidden_channels)

  def forward(self, x_dict, edge_index_dict, edge_label_index):
    z_dict = self.encoder(x_dict, edge_index_dict)
    out = self.decoder(z_dict, edge_label_index)
    return z_dict, out


def shuffle_label_data(train_data, a=('drug', 'interaction', 'protein')):

  length = train_data[a].edge_label.shape[0]
  lff = list(np.arange(length))
  random.shuffle(lff)

  train_data[a].edge_label = train_data[a].edge_label[lff]
  train_data[a].edge_label_index = train_data[a].edge_label_index[:, lff]
  return train_data
