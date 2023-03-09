import os
import pandas as pd
import argparse

import logging

from utils import get_sequences, seq2rat, get_pubchem2_smiles_in_batch


def main():
  """
  Fetch the drug data from pubchem.
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

  ds = args.database.lower()
  print('data from: ', ds)

  # Load dti data in the format of pubchem_uniprot
  logging.info(f'Working with {ds}')

  dti_file_path = f'../../data/raw/{ds}_dtis_pubchem_uniprot.tsv'
  dti = pd.read_csv(dti_file_path, sep='\t')
  dti['Drug'] = dti.Drug.astype(str)

  # Output path
  OUTPUT_PATH = f'../../data/{ds.upper()}'
  if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

  # Generate features for drugs
  drugs_unique = dti.Drug.unique().tolist()
  pub2smiles = get_pubchem2_smiles_in_batch(drugs_unique, size=25)

  drug_features = pd.DataFrame(drugs_unique, columns=['PubChem'])
  drug_features['PubChem'] = drug_features.PubChem.astype(str)

  drug_features['smiles'] = drug_features.PubChem.map(pub2smiles)
  drug_features = drug_features.dropna()

  drug_features.to_csv(
      '../KPGT/datasets/nr/nr.csv')

if __name__ == "__main__":
  main()
