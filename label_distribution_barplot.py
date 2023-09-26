import os
import numpy as np
import pandas as pd
import h5py
import pickle
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

file_marker_dir = 'data/file_markers_dodh/processed/25_nas/fold0'

k = 0
for i in os.listdir(file_marker_dir):
    p_num = {}
    p = []
    if i!='patient_split.csv':
        print(i)
        file_marker = pd.read_csv(os.path.join(file_marker_dir, i))
        for idx, f in enumerate(file_marker['record_id']):
            p_id = f.split('_')[0]
            flag = False
            if p_id not in p:
                p.append(p_id)
                flag=True
            label = file_marker['label'][idx]
            try:
                p_num[label].append(label)
            except:
                p_num[label] = []
                p_num[label].append(label)

        sns.set_theme(style='whitegrid')
        for fold in p_num.keys():
            x = list(p_num.keys())
            y = []
            print(x)
            for f in x:
                y.append(Counter(p_num[f])[f])
            print(y)
        plt.figure()
        sns.barplot(x=x, y=y)
        plt.show()
        plt.savefig('barplot_'+file_marker_dir.split('/')[-2].split('_')[0]+'_'+file_marker_dir.split('/')[-1]+'_'+i.split('_')[0]+'.png')




    