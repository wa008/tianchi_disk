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
        # if day != 201807: continue
        df = read_data_csv(data_pre_name + str(day), read_rows)
        cnt = df.shape
        for col in ['manufacturer', 'model', 'serial_number']: df[col] = df[col].astype(str)
        df = df.merge(df_label, on = ['manufacturer', 'model', 'serial_number'], how = 'left')
        df_positive = df[~df['tag'].isna()]
        df = df.sample(frac = 0.1, random_state = 2020)
        df_negative = df[df['tag'].isna()]
        df_negative = df_negative.sample(n = len(df_positive) * 5, random_state = 2020)
        df = pd.concat([df_positive, df_negative])
        df.to_csv(data_path + data_pre_name + str(day) + '_sample_pn.csv', index = False)
        print day, cnt, len(df_positive), len(df)
    return df

def read_data():
    read_rows = int(sys.argv[1])
    data_pre_name = 'disk_sample_smart_log_'
    df = read_data_csv(data_pre_name + '201707' + '_sample_pn', read_rows)
    print '201707', df.shape
    for day in range(201708, 201713) + range(201801, 201808):
        df_temp = read_data_csv(data_pre_name + str(day) + '_sample_pn', read_rows)
        df = pd.concat([df, df_temp])
        print day, df.shape
    df_test = read_data_csv('disk_sample_smart_log_test_a', -1)
    return df, df_test

def get_weight_label(df_train, df_test):
    print '\nget_weight_label', '-' * 100
    print 'dt, fault_time isna : ', df_train['dt'].isna().sum(), df_train['fault_time'].isna().sum()
    # print df_train[['dt', 'fault_time']].head(100)
    df_train['dt'] = df_train['dt'].fillna('19970102')
    df_train['fault_time'] = df_train['fault_time'].fillna('1997-01-02')
    df_train['dt_day'] = df_train['dt'].apply(lambda x: int(time.mktime(time.strptime(str(x), "%Y%m%d")) / 24 / 3600))
    df_train['broken_day'] = df_train['fault_time'].apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d")) / 24 / 3600))
    df_train['dis_day'] = (df_train['broken_day'] - df_train['dt_day']).apply(lambda x: -1 if x < 0 else -1 if x >= 30 else x)
    # print df_train[df_train['dt'] == 20170709][['dt', 'fault_time', 'broken_day', 'dt_day', 'dis_day']].head(10)
    df_train['label'] = df_train['dis_day'].apply(lambda x: 1 if x != -1 else 0)
    weight = df_train['dis_day'].apply(lambda x: 1 if x == -1 else 30 - x).values
    label = df_train['label'].values
    return df_train, df_test, weight, label

def data_process(df_train, df_test, cols, col_ratio = 0.6):
    print '\ndata_process', '-' * 100
    df = pd.concat([df_train, df_test])
    for col in cols:
        if col not in df_train.columns.tolist():
            print 'df_trian error', col
        if col not in df_test.columns.tolist():
            print 'df_test error', col
        if col not in df.columns.tolist():
            print 'df error', col
    drop_cols = []
    for col in cols:
        tmp = df[col].isna().sum() * 1.0 / len(df)
        if tmp > col_ratio:
            drop_cols.append(col)
    print 'drop_col_ratio : %.3f, num of drop cols : %d last cols : %d' % (col_ratio, len(drop_cols), len(cols) - len(drop_cols))
    df = df.drop(drop_cols, axis = 1)
    for col in drop_cols:
        cols.remove(col)
    for col in cols:
        df[col] = df[col].astype(np.float64)
        df[col] = df[col].fillna(df[col].mean())

    df_train = df.iloc[: len(df_train), :]   
    df_test = df.iloc[len(df_train) : , :]   
    return df_train, df_test, cols

def train(df_train, df_test, cols, weight, label):
    print '\ntrain' + '-' * 100
    print Counter(list(label))
    lgbm = lgb.LGBMClassifier()
    lgbm.fit(X = df_train[cols], y = label, sample_weight = weight)
    pred = lgbm.predict(df_test[cols])
    df_test['pred'] = pred
    print 'test Counter :', Counter(df_test['pred'].tolist())
    return df_test

def main():
    pre_time = time.time()
    # take_sample()    
    df_train, df_test = read_data()
    print 'df_train.shape, df_test.shap : ', df_train.shape, df_test.shape
    cols = df_test.columns.tolist()
    key_cols = ['manufacturer', 'model', 'serial_number', 'dt']
    for col in key_cols:
        if col in cols: cols.remove(col)
    df_trian, df_test, weight, label = get_weight_label(df_train, df_test)
    print 'sum of weight : %d' % np.sum(weight)

    df_train = df_train[key_cols + cols]
    df_test = df_test[key_cols + cols]


    df_trian, df_test, cols = data_process(df_trian, df_test, cols, col_ratio = 0.6)
    df_result = train(df_trian, df_test, cols, weight, label)
    df_result = df_result[df_result['pred'] == 1]
    df_result['dt'] = df_result['dt'].apply(lambda x: time.strftime('%Y-%m-%d', time.strptime(str(x), '%Y%m%d')))
    df_result[['manufacturer', 'model', 'serial_number', 'dt']].to_csv(data_path + 'sub_20200229.csv', index = False, header = False)
    pass
    print 'sepend time : ', time.time() - pre_time

main()
