import os
import glob
import pandas as pd

filedir = './raw_data/'
outdir = './extracted_data/'

fields = ['Field (Oersted)', 'Sample Temp (Kelvin)', 'Samp HC (µJ/K)', 'Samp HC Err (µJ/K)']

for filepath in glob.glob(os.path.join(filedir, '*.dat')):
    filename = filepath[len(filedir):-4]
    columntitlerow = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if line == '[Data]\n':
                columntitlerow = i + 1
    data = pd.read_csv(filepath, skiprows = list(range(columntitlerow)), encoding='unicode_escape')
    extracted_data = data[fields].dropna()
    
    extracted_data.to_pickle(os.path.join(outdir, f'{filename}.pkl'))