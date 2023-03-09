from model.model import Model
import os
import logging
import torch
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.append('../')


@torch.no_grad()
def test(model, data):
  model.eval()
  emb, pred = model(data.x_dict, data.edge_index_dict,
                    data['drug', 'protein'].edge_label_index)

  # The target value
  target = data['drug', 'protein'].edge_label.float()

  out = pred.view(-1).sigmoid()
  auc = roc_auc_score(target.cpu().numpy(), out.detach().cpu().numpy())
  aupr = average_precision_score(
      target.cpu().numpy(), out.detach().cpu().numpy())

  return round(auc, 6), round(aupr, 6), out, target


def sh_evaluation(ds_trained, ds_eval, hd=8):
  """
  Evaluate using a non-overlapping dataset against the trained model
  """

  # Load model
  logging.debug('------> Loading the model...')
  device = 'cuda'

  if ds_trained != ds_eval:
    fullname_trained = f'{ds_trained}_WO_{ds_eval}'.lower()
  else:
    fullname_trained = ds_trained

  model_path = os.path.join(
      f'../../results', f'{fullname_trained.upper()}_{hd}', 'model.pt')

  print(f'Using trained: {fullname_trained}')

  eval_data_path = f'../../data/{ds_eval.upper()}/hetero_data_{ds_eval.lower()}.pt'

  logging.debug('----> Loading Data (trained)...')

  # Load the evaluation data
  data_eval = torch.load(eval_data_path)

  # Prepare for the graph data object
  data_eval = T.ToUndirected()(data_eval)
  del data_eval['protein', 'rev_interaction', 'drug'].edge_label
  logging.debug('----> Loading the model...')

  # Load the model
  model = Model(hidden_channels=hd, data=data_eval).to(device)

  model.load_state_dict(torch.load(model_path))
  model.eval()

  # Process data for evaluation, including adding random negatives
  split_val = T.RandomLinkSplit(
      num_val=0.0,
      num_test=0.0,
      is_undirected=True,
      add_negative_train_samples=True,
      neg_sampling_ratio=1.0,
      edge_types=[('drug', 'interaction', 'protein')],
      rev_edge_types=[('protein', 'rev_interaction', 'drug')],
      split_labels=False
  )

  logging.debug('---> Splitting eval')
  sp_data, _, _ = split_val(data_eval)

  logging.debug('---> Testing eval')
  auc, aupr, _, _ = test(model, sp_data)

  print(
      f'Trained in {ds_trained}, evaluating in {ds_eval}, AUC: {auc}, AUPR: {aupr}')

  return auc, aupr
