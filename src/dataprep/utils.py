import os
import numpy as np
import requests
from collections import Counter
from rdkit import Chem
from tqdm import tqdm
from time import sleep
import logging
import matplotlib.pyplot as plt
import seaborn as sns


def seq2rat(sequence):
  """
  A simple approach to generate some initial embeddings for protein    
  """

  dict_count = dict(Counter(sequence))
  len_seq = len(sequence)

  list_aminos = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
                 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                 'T', 'W', 'Y', 'V']

  list_aminos.sort()

  out = [round(dict_count.get(amino, 0)/len_seq, 5) for amino in list_aminos]
  return out


def get_sequences(protein_list):
  """
  Fetch protein data from uniprot
  """
  dict_protein_seq = {}

  # slice if large
  if len(protein_list) > 500:
    n = 400
    protein_list = [protein_list[i:i + n]
                     for i in range(0, len(protein_list), n)]
  else:
    n = int(len(protein_list)/2)
    protein_list = [protein_list[i:i + n]
                     for i in range(0, len(protein_list), n)]

  for l in tqdm(protein_list, desc='Retrieving uniprot sequence'):
    uniprot_list = ','.join(l)
    r = requests.get(
        f'https://rest.uniprot.org/uniprotkb/accessions?accessions={uniprot_list}')
    jsons = r.json()['results']
    for data in tqdm(jsons, desc='saving to dict'):
      name = data.get('primaryAccession')
      res = data.get('sequence').get('value')
      dict_protein_seq[name] = res
    if len(protein_list) > 50:
      sleep(1)

  return dict_protein_seq


def get_pubchem2_smiles_in_batch(drugs, size=500):
  """
  Fetch drug data from pubchem
  """
  pub2smiles = {}

  # Split the list in chunks of 100 to make requests
  drugs = [str(drug) for drug in drugs]

  def split_list(big_list, x): return [
      big_list[i: i + x] for i in range(0, len(big_list), x)
  ]

  drug_chunks = split_list(drugs, size)
  for chunk in tqdm(drug_chunks, desc='Requesting SMILES to PubChem'):
    chunk = ",".join(chunk)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{chunk}/json"
    response = requests.get(url)
    if response.status_code == 200:
      jsons = response.json()
      for id in jsons.get("PC_Compounds"):
        cid, smile, inchikey = None, None, None
        cid = str(id.get('id').get('id').get('cid'))
        smile = [prop.get('value').get('sval') for prop in id.get('props')
                 if prop.get('urn').get('label') == 'SMILES'
                 and prop.get('urn').get('name') == 'Canonical'][0]

        if smile:
          try:
            mol1 = Chem.MolFromSmiles(str(smile))
            fp1 = Chem.RDKFingerprint(mol1)
          except:
            logging.info(f'Error for pubchemid {cid}')
            smile = None
        pub2smiles[cid] = smile

  return pub2smiles


def plot_auc(ds, output_path, results, hidden_channels, evtype):
  print('Plotting loss over epochs')
  plt.clf()
  plt.title(
      f'{evtype.upper()} {ds.upper()} - hidden_channels: {hidden_channels}\n train {results[0][-1]:.4f} test {results[2][-1]:.4f}')
  plt.ylabel('AUC')
  plt.xlabel('Epochs')
  plt.plot(results[0], 'k', label='train')
  plt.plot(results[1], 'purple', label='val')
  plt.plot(results[2], 'r', label='test')

  if evtype == 'loss':
    plt.legend(loc='upper right')
  else:
    plt.legend(loc='lower right')

  plt.savefig(os.path.join(
      output_path, f'ev_{evtype}_{ds.lower()}.png'), dpi=300)


def plot_heatmap(df, df_std, cmap='Reds'):
  # Create an array to annotate the heatmap
  labels = ["{0:.4f}\n$\pm$\n{1:.4f}".format(symb, value) for symb, value in zip(
      df.values.flatten(), df_std.values.flatten())]
  labels = np.asarray(labels).reshape(df.shape)
  plt.clf()
  fig = plt.figure()
  fig.set_size_inches(10, 10)
  ax = sns.heatmap(df, annot=labels, fmt="", vmin=0.50, vmax=1.0, cmap=cmap)
  ax.set(xlabel="Trained", ylabel="Evaluated")
  ax.xaxis.tick_top()
