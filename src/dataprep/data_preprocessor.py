import os
import pandas as pd
import json
import argparse
import pickle
import torch
import esm
from torch_geometric.data import HeteroData
import numpy as np

import logging

from utils import get_sequences, seq2rat, get_pubchem2_smiles_in_batch


def main():
  """
  Runs the data pre-processing including the pretrained model of ESM, KPGT and so on.
  Also stores the output checkpoint model data file.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--verbose", dest="verbosity", action="count", default=3,
                      help="Verbosity (between 1-4 occurrences with more leading to more "
                      "verbose logging). CRITICAL=0, ERROR=1, WARN=2, INFO=3, "
                      "DEBUG=4")
  parser.add_argument("-d", "--database",
                      help="database: e, nr, ic, gpcr, drugbank", type=str)

  args = parser.parse_args()
  log_levels = {
      0: logging.CRITICAL,
      1: logging.ERROR,
      2: logging.WARN,
      3: logging.INFO,
      4: logging.DEBUG,
  }

  # Set the logging level
  level = log_levels[args.verbosity]
  fmt = '[%(levelname)s] %(message)s]'
  logging.basicConfig(format=fmt, level=level)

  dataset = args.database.lower()
  print('data from: ', dataset)

  # Loading dti in format pubchem_uniprot
  logging.info(f'Working with {dataset}')

  dti_file_path = f'../../data/raw/{dataset}_dtis_pubchem_uniprot.tsv'
  dti = pd.read_csv(dti_file_path, sep='\t')
  dti['Drug'] = dti.Drug.astype(str)

  # Output path
  output_path = f'../../data/{dataset.upper()}'
  if not os.path.isdir(output_path):
    os.makedirs(output_path)

  drugs_unique = dti.Drug.unique().tolist()
  pub2smiles = get_pubchem2_smiles_in_batch(drugs_unique, size=25)

  drug_features = pd.DataFrame(drugs_unique, columns=['PubChem'])
  drug_features['PubChem'] = drug_features.PubChem.astype(str)

  drug_features['SMILES'] = drug_features.PubChem.map(pub2smiles)
  drug_features = drug_features.dropna()
  available_drugs = drug_features.PubChem.unique().tolist()

  drug_re = np.load('../KPGT/datasets/nr/kpgt_base.npz')['fps']

  # Generate features with RDKit
  drug_x = torch.from_numpy(drug_re).to(torch.float)

  # Generate features for proteins
  proteins_unique = dti.Protein.unique().tolist()

  protein_feaures = pd.DataFrame(proteins_unique, columns=['Uniprot'])
  logging.debug('Retrieving sequences...')
  prot2seq = get_sequences(proteins_unique)
  protein_feaures['Sequence'] = protein_feaures.Uniprot.map(prot2seq)
  protein_feaures = protein_feaures.dropna()
  available_proteins = protein_feaures.Uniprot.unique().tolist()

  # Load ESM pretrained model
  print("processing protein ESM-1b")
  model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
  model = model.cpu()
  batch_converter = alphabet.get_batch_converter()
  model.eval()
  data_protein = []
  for rows in protein_feaures.iterrows():
    if len(rows[1][1]) > 1022:
      data_protein.append((rows[1][0], rows[1][1][:1022]))
    else:
      data_protein.append((rows[1][0], rows[1][1]))

  sequence_representations = []
  for j in range(0, len(data_protein), 1):
    data_2 = data_protein[j:j+5]

    batch_labels, batch_strs, batch_tokens = batch_converter(data_2)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
      results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
      sequence_representations.append(
          token_representations[i, 1: tokens_len - 1].mean(0))

  protein_x = torch.stack(sequence_representations).to(torch.float)

  # Generate features for drugs
  new_df = dti[dti.Drug.isin(available_drugs) &
               dti.Protein.isin(available_proteins)]
  new_df = new_df.drop_duplicates(keep='first')

  # Creating a new df for available drug and protein pairs
  new_df = dti[dti.Drug.isin(available_drugs) &
               dti.Protein.isin(available_proteins)]

  new_df = new_df.drop_duplicates(keep='first')

  df_disease = pd.read_csv('../../data/raw/gene.dis.csv')
  with open('../../data/pro2gene.pickle', 'rb') as handle:
    pro2gene_dict = pickle.load(handle)
  disease_list = []
  for i in new_df['Protein'].tolist():
    try:
      h = pro2gene_dict[i]
      if h in df_disease['x_name'].tolist():
        x_data = df_disease[df_disease['x_name'] == h]
        x_data['uni'] = i
        disease_list.append(x_data)
    except:
      pass

  df_disease_tmp = pd.concat(disease_list)
  with open('../../data/disease_embd.pickle', 'rb') as handle:
    di_embed = pickle.load(handle)
  df_disease = pd.DataFrame(df_disease_tmp, columns=['y_name', 'uni'])
  di_unique = df_disease.y_name.unique()
  di_features = pd.DataFrame(di_unique, columns=['Dise'])
  di_embed_l = []
  for d in di_unique:
    di_embed_l.append(torch.from_numpy(di_embed[d]))

  print(f'New shape after retrieven features: {new_df.shape}')
  print(f'Unique drugs: {len(new_df.Drug.unique())}')
  print(f'Unique proteins: {len(new_df.Protein.unique())}')
  print(f'Unique Disease: {len(df_disease.y_name.unique())}')

  # Make sure that all drugs are in df if not drop row
  drug_features = drug_features[drug_features.PubChem.isin(
      new_df.Drug)].reset_index(drop=True)
  protein_feaures = protein_feaures[protein_feaures.Uniprot.isin(
      new_df.Protein)].reset_index(drop=True)

  # Prepare for the hetero data
  drug_mapping = {index: i for i, index in enumerate(
      drug_features.PubChem.tolist())}

  protein_mapping = {index: i for i, index in enumerate(
      protein_feaures.Uniprot.tolist())}

  disease_x = torch.stack(di_embed_l, dim=0)
  disease_mapping = {index: i for i, index in enumerate(list(di_unique))}

  src = [drug_mapping[index] for index in new_df['Drug']]
  dst = [protein_mapping[index] for index in new_df['Protein']]
  edge_index = torch.tensor([src, dst])

  src1 = [protein_mapping[index] for index in df_disease['uni']]
  dst1 = [disease_mapping[index] for index in df_disease['y_name']]
  edge_index1 = torch.tensor([src1, dst1])

  # Generate the graph data object
  data = HeteroData()
  data['drug'].x = drug_x
  data['protein'].x = protein_x
  data['disease'].x = disease_x
  data['drug', 'interaction', 'protein'].edge_index = edge_index
  data['protein', 'asso', 'disease'].edge_index = edge_index1
  print(data)

  device = 'cuda:1'
  data = data.to(device)

  # Save output files
  logging.info(f'Saving files in {output_path}')

  with open(os.path.join(output_path, 'drug_mapping.json'), "w") as outfile:
    json.dump(drug_mapping, outfile)

  with open(os.path.join(output_path, 'protein_mapping.json'), "w") as outfile:
    json.dump(protein_mapping, outfile)

  with open(os.path.join(output_path, 'di_mapping.json'), "w") as outfile:
    json.dump(disease_mapping, outfile)

  # Save the datafile
  torch.save(data, os.path.join(
      output_path, f'hetero_data_{dataset.lower()}.pt'))

  # Save the df
  new_df.to_pickle(os.path.join(output_path, f'dti_{dataset.lower()}.pkl'))

  with open(os.path.join(output_path, f'info_dti_{dataset.lower()}.out'), "w") as f:
    f.write(f'New shape after retrieven features: {new_df.shape}\n')
    f.write(f'Unique drugs: {len(new_df.Drug.unique())}\n')
    f.write(f'Unique proteins: {len(new_df.Protein.unique())}\n')


if __name__ == "__main__":
  main()
