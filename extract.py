import os
import pandas as pd
import numpy as np
import pickle as pkl

clean_folder_path = './data/2_clean'
data_folder = 'near_T2'
data_folder_path = os.path.join(clean_folder_path, data_folder)

extracted_folder_path = './data/3_extracted'
save_name = data_folder + '.pkl'
extracted_file_path = os.path.join(extracted_folder_path, save_name)

def float_eval(string):
    try:
        return float(string)
    except:
        return string

data = dict()
for file in os.listdir(data_folder_path):
    file_path = os.path.join(data_folder_path, file)
    if file.endswith('.dat'):
        with open(file_path) as f:
            lines = []
            for line in f.readlines():
                lines.append(list(map(float_eval, list(map(str.strip, line.split(','))))))
        
        keys = [key.split()[0] for key in lines[0]]
        valuess = lines[1:]
        datum = [dict(zip(keys, values)) for values in valuess]

        datum = pd.DataFrame(datum)

        B = int(np.mean(datum['Field'])/10000)
        data[B] = datum
    elif file.endswith('.xlsx'):
        datum = pd.read_excel(file_path)
        B = file[5:-(len(data_folder) + 7)]
    
with open(extracted_file_path, 'wb') as f:
    pkl.dump(data, f)