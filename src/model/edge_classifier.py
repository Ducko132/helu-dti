import torch
from torch.nn import Linear


class EdgeClassifier(torch.nn.Module):
  """
  Implements the Multi-Layer Perceptron neural network to predict the drug and target affinity
  """

  def __init__(self, hidden_channels):
    super().__init__()
    self.lin1 = Linear(2 * 128, 64)
    self.lin2 = Linear(64, 1)

    self.drug_lin1 = Linear(hidden_channels, 2*hidden_channels)
    self.drug_lin2 = Linear(2*hidden_channels, 128)

    self.pro_lin1 = Linear(hidden_channels, 2*hidden_channels)
    self.pro_lin2 = Linear(2*hidden_channels, 128)

  def forward(self, z_dict, edge_label_index):
    row, col = edge_label_index
    z_d = self.drug_lin1(z_dict['drug'][row]).relu()
    z_d = self.drug_lin2(z_d).relu()

    z_p = self.pro_lin1(z_dict['protein'][col]).relu()
    z_p = self.pro_lin2(z_p).relu()

    z = torch.cat([z_d, z_p], dim=-1)
    z = self.lin1(z).relu()
    z = self.lin2(z)
    return z.view(-1)
