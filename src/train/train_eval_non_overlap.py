from .eval import sh_evaluation
from dataprep.utils import plot_heatmap
import argparse
import logging
import subprocess as sp

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

import sys
sys.path.append('../')


"""
Train on all datasets and perform generalized evaluation based on non-overlapping datasets
"""

hidden_channels = 7

print(f'Embedding dimension {hidden_channels}')

# Results folder
RESULTS_FOLDER = '../../results/'

# All datasets used in train and test
datasets_1 = ['DrugBank', 'BIOSNAP',
              'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR']
datasets_2 = datasets_1

nreps_training = 5
nreps = 5

for r in tqdm(range(nreps_training), desc='Repetitions'):
  for ds1 in datasets_1:

    print('======================')
    print(f'Working with {ds1}')

    for ds2 in datasets_2:
      if ds2 != ds1:
        ds = f'{ds1}' + '_WO_' + f'{ds2}'
        print(ds)

        try:
          print(f'database {ds}, hidden_channels {hidden_channels}')
          ret_code = sp.check_call(
              ['python3', 'train.py', '-d', f'{ds}', '-e', f'{hidden_channels}'])
          if ret_code == 0:
            print(
                f'EXIT CODE 0 FOR {ds.upper()} with hd {hidden_channels}')

        except sp.CalledProcessError as e:
          logging.info(e.output)

      else:
        ds = f'{ds1}'
        print(ds)

        try:
          print(f'database {ds}, hidden_channels {hidden_channels}')
          ret_code = sp.check_call(
              ['python3', 'train.py', '-d', f'{ds}', '-e', f'{hidden_channels}'])
          if ret_code == 0:
            print(
                f'EXIT CODE 0 FOR {ds.upper()} with hd {hidden_channels}')

        except sp.CalledProcessError as e:
          logging.info(e.output)

    print(f'Completed with {ds1}')

  df_auc_hmp = pd.DataFrame(columns=datasets_1, index=datasets_2).astype(float)
  df_aupr_hmp = pd.DataFrame(
      columns=datasets_1, index=datasets_2).astype(float)

  for rep in range(nreps):
    for dataset_trained in datasets_1:
      for dataset_evaluated in datasets_2:
        auc_list = []
        aupr_list = []
        auc, aupr = sh_evaluation(
            ds_trained=dataset_trained, ds_eval=dataset_evaluated,  hd=hidden_channels)
        df_auc_hmp[dataset_trained].loc[dataset_evaluated] = auc
        df_aupr_hmp[dataset_trained].loc[dataset_evaluated] = aupr

    print(f"==== {hidden_channels} ====")

    print('AUC')
    print(df_auc_hmp)

    print('AUPR')
    print(df_aupr_hmp)

    # Save data
    if not os.path.isdir(RESULTS_FOLDER):
      os.makedirs(RESULTS_FOLDER)
    df_auc_hmp.to_pickle(
        f'{RESULTS_FOLDER}tmp_auc_{hidden_channels}_{r}_{rep}.pkl')
    df_aupr_hmp.to_pickle(
        f'{RESULTS_FOLDER}tmp_aupr_{hidden_channels}_{r}_{rep}.pkl')


print('Completed!')


df_auc_hmp = pd.DataFrame(columns=datasets_1, index=datasets_2).astype(float)
df_aupr_hmp = pd.DataFrame(columns=datasets_1, index=datasets_2).astype(float)

df_auc_std_hmp = pd.DataFrame(
    columns=datasets_1, index=datasets_2).astype(float)
df_aupr_std_hmp = pd.DataFrame(
    columns=datasets_1, index=datasets_2).astype(float)

for dataset_trained in datasets_1:
  for dataset_evaluated in datasets_2:
    auc_list, aupr_list = [], []
    for r in range(nreps_training):
      for rep in range(nreps):
        (r, rep)

        # AUC
        df_auc = pd.read_pickle(
            f'{RESULTS_FOLDER}tmp_auc_{hidden_channels}_{r}_{rep}.pkl')
        auc = df_auc[dataset_trained].loc[dataset_evaluated]
        auc_list.append(auc)

        # AUPR
        df_aupr = pd.read_pickle(
            f'{RESULTS_FOLDER}tmp_aupr_{hidden_channels}_{r}_{rep}.pkl')
        aupr = df_aupr[dataset_trained].loc[dataset_evaluated]
        aupr_list.append(aupr)

    final_auc = np.array(auc_list).mean()
    final_auc_std = np.array(auc_list).std()
    final_aupr = np.array(aupr_list).mean()
    final_aupr_std = np.array(aupr_list).std()

    # Add heatmap
    df_auc_hmp[dataset_trained].loc[dataset_evaluated] = final_auc
    df_aupr_hmp[dataset_trained].loc[dataset_evaluated] = final_aupr
    df_auc_std_hmp[dataset_trained].loc[dataset_evaluated] = final_auc_std
    df_aupr_std_hmp[dataset_trained].loc[dataset_evaluated] = final_aupr_std


# Save data
df_auc_hmp.to_pickle(f'{RESULTS_FOLDER}auc_{hidden_channels}.pkl')
df_aupr_hmp.to_pickle(f'{RESULTS_FOLDER}aupr_{hidden_channels}.pkl')
df_auc_std_hmp.to_pickle(f'{RESULTS_FOLDER}auc_std_{hidden_channels}.pkl')
df_aupr_std_hmp.to_pickle(f'{RESULTS_FOLDER}aupr_std_{hidden_channels}.pkl')

# Plot heatmap
plot_heatmap(df_auc_hmp, df_auc_std_hmp, cmap='Oranges')
plt.savefig(f'{RESULTS_FOLDER}results_AUC_{hidden_channels}_.pdf',
            bbox_inches='tight',  pad_inches=0.2)
plot_heatmap(df_aupr_hmp, df_aupr_std_hmp, 'Oranges')
plt.savefig(f'{RESULTS_FOLDER}results_AUPR_{hidden_channels}_.pdf',
            bbox_inches='tight',  pad_inches=0.2)
