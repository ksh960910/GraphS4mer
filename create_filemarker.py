import pandas as pd
import csv
import numpy as np
import os 
import re
import pickle
from tqdm import tqdm

match_label = {'SLEEP-S0' : 0, 'SLEEP-S1' : 1, 'SLEEP-S2' : 2, 'SLEEP-S3' : 3, 'SLEEP-REM' : 4}

# Sound data npy file에서 저장된 label 불러오기
sound_data_dir = '/nas/max/tmp_data/dataset_abcd/psg_abc/'
psg_data_dir_12 = '/nas/SNUBH-PSG_signal_extract/signal_extract_12'

# # K-Fold로 나누기
# # CHIN EMG가 있는 patient
# full_patient = [1160,989,1244,1180,1141,1433,1027,1435,1137,1151,1049,1047,1085,1058,1018,1441,1360,1263,970,1481,1452,1337,1273,1117,1007, 1479,1045,1400,997,1258,1368,1079,1299,1423,1469,1317,1455,1201,1066,1349]

# print('length :' ,len(full_patient))

# k = 0
# if 5*k+15>len(full_patient):
#     train_num = full_patient[5*k:]+full_patient[:5*k+15-len(full_patient)]
#     left = [x for x in full_patient if x not in train_num]
#     print(left)
#     if (k+1)%2==0:
#         val_num = left[5:10]
#         test_num = left[:5]
#     elif (k+1)%2==1:
#         val_num = left[:5]
#         test_num = left[5:10]
# else:
#     train_num = full_patient[5*k:5*k+15]
#     left = [x for x in full_patient if x not in train_num]
#     print(left)
#     if (k+1)%2==0:
#         val_num = left[5:10]
#         test_num = left[:5]
#     elif (k+1)%2==1:
#         val_num = left[:5]
#         test_num = left[5:10]
# print(train_num)
# print(val_num)
# print(test_num)

def get_patient_id(psg_data_dir, group, mode):
    # 각 train/val/test에 해당할 patient_id 뽑아서 각각 array에 저장
    patient_list = []
    sorted_patient = []
    path = os.path.join(psg_data_dir, group, mode)
    for i in os.listdir(path):
        patient_list.append(i.split('_')[0])
    patient_list = list(set(patient_list))

    print('Check each length : ', len(train_num), len(val_num), len(test_num))
    return train_num, val_num, test_num


def convert_label(sound_data_dir, psg_data_dir, group, mode='train'):
    file_markers = pd.DataFrame()
    if mode=='train':
        for num in tqdm(train_num):
            # Select the npy files that match patient number from sound_data_dir 
            fname = []
            pattern = re.escape(str(num))
            for f in os.listdir(os.path.join(psg_data_dir,group,'train')):
                if re.match(pattern, f)!=None and len(pattern)==len(f.split('_')[0]):
                    fname.append(f)
            # Sort it in order : First by disconnection number, Second by session number
            fname.sort(key=lambda x : (int(x.split('_')[2]), int(x.split('_')[-1].split('.')[0])))
            '''Open each npy file and get the value corresponding to key '5c' 
                and save the pkl file name from psg_data_dir to record_id
                and save the session number to clip_index'''
            # file markers
            marker = pd.DataFrame()
            label = []
            match_fname = []
            for name in fname:
                try:
                    data = np.load(sound_data_dir+group+'/train/'+name.split('.')[0]+'.npy', allow_pickle=True)
                    match_fname.append(group+'/train/'+name)
                    label.append(int(data.tolist()['5c']))
                except:
                    continue
            marker['record_id'] = list(map(lambda x : x.split('.')[0]+'.pkl', match_fname))
            marker['clip_index'] = range(len(match_fname))
            marker['label'] = label
            marker['seq_len'] = np.repeat(30 * 500, len(match_fname))
            file_markers = pd.concat([file_markers, marker], ignore_index=True)
        file_markers.to_csv('data/file_markers_dodh/processed/all_12ch/train_file_markers.csv', mode='a', sep=',', index=False)
    
    elif mode=='val':
        for num in tqdm(val_num):
            # Select the npy files that match patient number from sound_data_dir 
            fname = []
            pattern = re.escape(str(num))
            for f in os.listdir(os.path.join(psg_data_dir,group,'test')):
                if re.match(pattern, f)!=None and len(pattern)==len(f.split('_')[0]):
                    fname.append(f)
            # Sort it in order : First by disconnection number, Second by session number
            fname.sort(key=lambda x : (int(x.split('_')[2]), int(x.split('_')[-1].split('.')[0])))
            '''Open each npy file and get the value corresponding to key '5c' 
                and save the pkl file name from psg_data_dir to record_id
                and save the session number to clip_index'''
            # file markers
            marker = pd.DataFrame()
            label = []
            match_fname = []
            for name in fname:
                try:
                    data = np.load(sound_data_dir+group+'/test/'+name.split('.')[0]+'.npy', allow_pickle=True)
                    match_fname.append(group+'/test/'+name)
                    label.append(int(data.tolist()['5c']))
                except:
                    continue
            marker['record_id'] = list(map(lambda x : x.split('.')[0]+'.pkl', match_fname))
            marker['clip_index'] = range(len(match_fname))
            marker['label'] = label
            marker['seq_len'] = np.repeat(30 * 500, len(match_fname))
            file_markers = pd.concat([file_markers, marker], ignore_index=True)
        file_markers.to_csv('data/file_markers_dodh/processed/all_12ch/val_file_markers.csv', mode='a', sep=',', index=False)

    if mode=='test':
        for num in tqdm(test_num):
            # Select the npy files that match patient number from sound_data_dir 
            fname = []
            pattern = re.escape(str(num))
            for f in os.listdir(os.path.join(psg_data_dir,group,'test')):
                if re.match(pattern, f)!=None and len(pattern)==len(f.split('_')[0]):
                    fname.append(f)
            # Sort it in order : First by disconnection number, Second by session number
            fname.sort(key=lambda x : (int(x.split('_')[2]), int(x.split('_')[-1].split('.')[0])))
            '''Open each npy file and get the value corresponding to key '5c' 
                and save the pkl file name from psg_data_dir to record_id
                and save the session number to clip_index'''
            # file markers
            marker = pd.DataFrame()
            label = []
            match_fname = []
            for name in fname:
                try:
                    data = np.load(sound_data_dir+group+'/test/'+name.split('.')[0]+'.npy', allow_pickle=True)
                    match_fname.append(group+'/test/'+name)
                    label.append(int(data.tolist()['5c']))
                except:
                    continue
            marker['record_id'] = list(map(lambda x : x.split('.')[0]+'.pkl', match_fname))
            marker['clip_index'] = range(len(match_fname))
            marker['label'] = label
            marker['seq_len'] = np.repeat(30 * 500, len(match_fname))
            file_markers = pd.concat([file_markers, marker], ignore_index=True)
        file_markers.to_csv('data/file_markers_dodh/processed/all_12ch/test_file_markers.csv', mode='a', sep=',', index=False)

def patient_split():
    patient = pd.DataFrame()
    recordlist, splitlist = [], []
    train_df = pd.read_csv('data/file_markers_dodh/processed/all_12ch/train_file_markers.csv')
    val_df = pd.read_csv('data/file_markers_dodh/processed/all_12ch/val_file_markers.csv')
    test_df = pd.read_csv('data/file_markers_dodh/processed/all_12ch/test_file_markers.csv')

    split = ['train']*len(train_df['record_id']) + ['val']*len(val_df['record_id']) + ['test']*len(test_df['record_id'])
    record_id = list(train_df['record_id']) + list(val_df['record_id']) + list(test_df['record_id'])
    patient['record_id'] = record_id
    patient['split'] = split

    patient.to_csv('data/file_markers_dodh/processed/all_12ch/patient_split.csv', sep=',', index=False)


for group in os.listdir(psg_data_dir):
    train_num, val_num, test_num = get_patient_id(psg_data_dir, group, mode='train')
    convert_label(sound_data_dir, psg_data_dir_12, group, mode='train')
    train_num, val_num, test_num = get_patient_id(psg_data_dir, group, mode='val')
    convert_label(sound_data_dir, psg_data_dir_12, group, mode='val')
    train_num, val_num, test_num = get_patient_id(psg_data_dir, group, mode='test')
    convert_label(sound_data_dir, psg_data_dir_12, group, mode='test')
patient_split()
