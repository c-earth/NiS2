import os
import time
import pandas as pd
import numpy as np
import pickle as pkl

extracted_folder_path = './data/3_extracted'
compiled_folder_path = './data/4_compiled'
save_name = time.strftime('%Y-%m-%d-%H-%M-%S') + '.pkl'
compiled_file_path = os.path.join(compiled_folder_path, save_name)

datas = dict()
for file in os.listdir(extracted_folder_path):
    file_path = os.path.join(extracted_folder_path, file)
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    datas[file] = data

with open(compiled_file_path, 'wb') as f:
    pkl.dump(datas, f)