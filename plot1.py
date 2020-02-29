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

def read_data():
    read_rows = int(sys.argv[1])
    if read_rows > 0:
        return read_data_csv('disk_sample_smart_log_201803', read_rows)
    data_pre_name = 'disk_sample_smart_log_'
    df = read_data_csv(data_pre_name + '201707')
    for day in range(201708, 201713) + range(201801, 201808):
        df_temp = read_data_csv(data_pre_name + str(day), -1)
        df = pd.concat([df, df_temp])
    return df

def data_process(df):
    temp = []
    for col in df.columns.values:
        temp.append([df[col].isna().sum() * 1.0 / len(df), col])
    temp = sorted(temp, reverse = True)
    for x in temp:
        print x[0], x[1]
    pass

def main():
    df = read_data()
    pass

main()