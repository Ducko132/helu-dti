import logging
import subprocess as sp

"""
Preprocess all pairs of datasets to generate non-overlapping datasets for train and test respectively
"""

datasets_1 = ['DrugBank', 'BIOSNAP',
              'BindingDB', 'Davis', 'E', 'GPCR', 'IC', 'NR']

datasets_2 = datasets_1

for ds1 in datasets_1:
  for ds2 in datasets_2:
    if ds2 != ds1:
      try:
        print(f'database 1 {ds1}, database 2 {ds2}')
        ret_code = sp.check_call(
            ['python3', 'data_preprocessor_multi.py', '-d', f'{ds1}', '-t', f'{ds2}'])
        if ret_code == 0:
          logging.info(
              f'Exit code 0 for {ds1.upper()} w\o {ds2.upper()}')

      except sp.CalledProcessError as e:
        logging.info(e.output)


print('0')
