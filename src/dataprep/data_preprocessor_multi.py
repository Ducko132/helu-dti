import os
import pandas as pd
from scipy.sparse import csr_matrix
import json
import argparse
import logging

from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
from torch_geometric.data import HeteroData


from utils import get_sequences, seq2rat, get_pubchem2_smiles_in_batch


def main():
  """
  Runs the pre-processing for non-overlapping datasets.
  Also stores the output checkpoint model data file.
  """

  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--verbose", dest="verbosity", action="count", default=3,
                      help="Verbosity (between 1-4 occurrences with more leading to more "
                      "verbose logging). CRITICAL=0, ERROR=1, WARN=2, INFO=3, "
                      "DEBUG=4")
  parser.add_argument("-d", "--database",
                      help="database: e, nr, ic, gpcr, drugbank", type=str)
  parser.add_argument("-t", "--databasetest",
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

  ds_train = args.database.lower()
  ds_test = args.databasetest.lower()

  print('data from: ', ds_train)

  # Loading the dti dataset
  logging.info(f'Working with {ds_train}')

  dti_train_data_path = f'../../data/raw/{ds_train}_dtis_pubchem_uniprot.tsv'

  dti = pd.read_csv(dti_train_data_path, sep='\t')
  dti['Drug'] = dti.Drug.astype(str)

  # Generate features for proteins
  proteins_unique = dti.Protein.unique().tolist()

  protein_feaures = pd.DataFrame(proteins_unique, columns=['Uniprot'])
  logging.debug('Retrieving sequences...')
  prot2seq = get_sequences(proteins_unique)
  protein_feaures['Sequence'] = protein_feaures.Uniprot.map(prot2seq)
  protein_feaures = protein_feaures.dropna()
  protein_feaures['Ratios'] = protein_feaures.Sequence.map(
      lambda x: seq2rat(x))

  available_proteins = protein_feaures.Uniprot.unique().tolist()
  logging.debug(protein_feaures.head(3))

  # Generate features for drugs
  drugs_unique = dti.Drug.unique().tolist()
  pub2smiles = get_pubchem2_smiles_in_batch(drugs_unique, size=25)

  drug_features = pd.DataFrame(drugs_unique, columns=['PubChem'])
  drug_features['PubChem'] = drug_features.PubChem.astype(str)

  drug_features['SMILES'] = drug_features.PubChem.map(pub2smiles)
  drug_features = drug_features.dropna()

  # Generate features with RDKit
  drug_features['MolLogP'] = drug_features.SMILES.map(
      lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x)))
  drug_features['MolWt'] = drug_features.SMILES.map(
      lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))
  drug_features['NumHAcceptors'] = drug_features.SMILES.map(
      lambda x: Descriptors.NumHAcceptors(Chem.MolFromSmiles(x)))
  drug_features['NumHDonors'] = drug_features.SMILES.map(
      lambda x: Descriptors.NumHDonors(Chem.MolFromSmiles(x)))
  drug_features['NumHeteroatoms'] = drug_features.SMILES.map(
      lambda x: Descriptors.NumHeteroatoms(Chem.MolFromSmiles(x)))
  drug_features['NumRotatableBonds'] = drug_features.SMILES.map(
      lambda x: Descriptors.NumRotatableBonds(Chem.MolFromSmiles(x)))
  drug_features['TPSA'] = drug_features.SMILES.map(
      lambda x: Descriptors.TPSA(Chem.MolFromSmiles(x)))
  drug_features['RingCount'] = drug_features.SMILES.map(
      lambda x: Descriptors.RingCount(Chem.MolFromSmiles(x)))
  drug_features['NHOHCount'] = drug_features.SMILES.map(
      lambda x: Descriptors.NHOHCount(Chem.MolFromSmiles(x)))
  drug_features['NOCount'] = drug_features.SMILES.map(
      lambda x: Descriptors.NOCount(Chem.MolFromSmiles(x)))
  drug_features['HeavyAtomCount'] = drug_features.SMILES.map(
      lambda x: Descriptors.HeavyAtomCount(Chem.MolFromSmiles(x)))
  drug_features['NumValenceElectrons'] = drug_features.SMILES.map(
      lambda x: Descriptors.NumValenceElectrons(Chem.MolFromSmiles(x)))

  logging.debug(drug_features.head())

  available_drugs = drug_features.PubChem.unique().tolist()

  # Remove duplicates
  dti_test_data_path = f'../../data/raw/{ds_test}_dtis_pubchem_uniprot.tsv'
  print(f'=================> USING TEST {dti_test_data_path}')
  dti_test = pd.read_csv(dti_test_data_path, sep='\t')

  dti_test['Drug'] = dti_test.Drug.astype(str)
  dti_test = dti_test.drop_duplicates(keep='first')

  on = ['Drug', 'Protein']
  dti = dti.merge(dti_test[on], on=on, how='left', indicator=True).query(
      '_merge == "left_only"').drop('_merge', 1)

  # Creating a new df with available drugs and proteins
  new_df = dti[dti.Drug.isin(available_drugs) &
               dti.Protein.isin(available_proteins)]
  new_df = new_df.drop_duplicates(keep='first')

  print(f'New shape after retrieven features: {new_df.shape}')
  print(f'Unique drugs: {len(new_df.Drug.unique())}')
  print(f'Unique proteins: {len(new_df.Protein.unique())}')

  # Make sure that all drugs are in df if not drop row
  drug_features = drug_features[drug_features.PubChem.isin(
      new_df.Drug)].reset_index(drop=True)
  protein_feaures = protein_feaures[protein_feaures.Uniprot.isin(
      new_df.Protein)].reset_index(drop=True)

  # Generate matrix of drug features
  drug_features_matrix = drug_features.drop(columns=['PubChem', 'SMILES'])

  drug_features_matrix = (drug_features_matrix-drug_features_matrix.min()) / \
      (drug_features_matrix.max()-drug_features_matrix.min())
  drug_features_sparse = csr_matrix(drug_features_matrix.values)

  # Generate matrix of protein features
  protein_feaures_matrix = pd.concat(
      [protein_feaures, protein_feaures.Ratios.apply(pd.Series)], axis=1)
  protein_feaures_matrix = protein_feaures_matrix.drop(
      columns=['Uniprot', 'Sequence', 'Ratios'])
  protein_feaures_sparse = csr_matrix(protein_feaures_matrix.values)

  # Generate the heterogeneous graph data object
  drug_x = torch.from_numpy(drug_features_sparse.todense()).to(torch.float)
  drug_mapping = {index: i for i, index in enumerate(
      drug_features.PubChem.tolist())}

  protein_x = torch.from_numpy(
      protein_feaures_sparse.todense()).to(torch.float)
  protein_mapping = {index: i for i, index in enumerate(
      protein_feaures.Uniprot.tolist())}

  src = [drug_mapping[index] for index in new_df['Drug']]
  dst = [protein_mapping[index] for index in new_df['Protein']]
  edge_index = torch.tensor([src, dst])

  # Generate the graph data object
  data = HeteroData()
  data['drug'].x = drug_x
  data['protein'].x = protein_x
  data['drug', 'interaction', 'protein'].edge_index = edge_index
  print(data)

  device = 'cuda'
  data = data.to(device)

  # Save output files
  output_path = f'../../data/{ds_train.upper()}_WO_{ds_test.upper()}'
  if not os.path.isdir(output_path):
    os.makedirs(output_path)

  logging.info(f'Saving files in {output_path}')

  with open(os.path.join(output_path, 'drug_mapping.json'), "w") as outfile:
    json.dump(drug_mapping, outfile)

  with open(os.path.join(output_path, 'protein_mapping.json'), "w") as outfile:
    json.dump(protein_mapping, outfile)

  # Save the data file
  torch.save(data, os.path.join(
      output_path, f'hetero_data_{ds_train.lower()}_wo_{ds_test.lower()}.pt'))

  # Save the df
  new_df.to_pickle(os.path.join(
      output_path, f'dti_{ds_train.lower()}_wo_{ds_test.lower()}.pkl'))

  with open(os.path.join(output_path, f'info_dti_{ds_train.lower()}_wo_{ds_test.lower()}.out'), "w") as f:
    f.write(f'New shape after retrieven features: {new_df.shape}\n')
    f.write(f'Unique drugs: {len(new_df.Drug.unique())}\n')
    f.write(f'Unique proteins: {len(new_df.Protein.unique())}\n')


if __name__ == "__main__":
  main()
