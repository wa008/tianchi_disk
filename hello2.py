#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
import json
import sys
import gc
from tqdm import tqdm
# machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_error, roc_auc_score
from collections import Counter
from sklearn.model_selection import GroupKFold
import xgboost
import lightgbm as lgb
import datetime
sys.path.append('..')
# myself libs
from read_data import read_data_csv, read_data_pkl
# other
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_path = r'D:\kaggle\data\tianchi_disk\a'[: -1]
print 'data_path :', data_path

def take_sample():
    read_rows = int(sys.argv[1])
    data_pre_name = 'disk_sample_smart_log_'
    df_label = pd.read_csv(data_path + 'disk_sample_fault_tag.csv', names = ['manufacturer', 'model', 'serial_number', 'fault_time', 'tag'])
    df_label = df_label.drop_duplicates()
    for col in ['manufacturer', 'model', 'serial_number']: df_label[col] = df_label[col].astype(str)
    print df_label.head(3)
    for day in range(201707, 201713) + range(201801, 201808):
        if day < 201807: continue
        df = read_data_csv(data_pre_name + str(day), read_rows)
        cnt = df.shape
        for col in ['manufacturer', 'model', 'serial_number']: df[col] = df[col].astype(str)
        df = df.merge(df_label, on = ['manufacturer', 'model', 'serial_number'], how = 'left')
        print 'debug1'
        df_positive = df[~df['tag'].isna()]
        print 'debug2', len(df_positive)
        df = df.sample(frac = 0.1, random_state = 2020)
        df_negative = df[df['tag'].isna()]
        print 'debug3'
        df_negative = df_negative.sample(n = len(df_positive) * 5, random_state = 2020)
        df = pd.concat([df_positive, df_negative])
        df.to_csv(data_path + data_pre_name + str(day) + '_sample_pn_copy.csv', index = False)
        print day, cnt, len(df_positive), len(df)
    return df

def main():
    take_sample()    
    pass

main()
