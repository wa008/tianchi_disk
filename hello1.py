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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score
from collections import Counter
from sklearn.model_selection import GroupKFold
import xgboost
import lightgbm as lgb
import datetime
# myself libs
from read_data import read_data_csv, read_data_pkl, write_data_csv
from validation import val_lgb
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
    df_label = df_label.groupby(['manufacturer', 'model', 'serial_number'])['fault_time'].min().to_frame().reset_index()
    print df_label.head(3)
    for col in ['manufacturer', 'model', 'serial_number']: df_label[col] = df_label[col].astype(str)
    for day in range(201707, 201713) + range(201801, 201808):
        # if day != 201807: continue
        df = read_data_csv(data_pre_name + str(day), read_rows)
        cnt = df.shape
        for col in ['manufacturer', 'model', 'serial_number']: df[col] = df[col].astype(str)
        df = df.merge(df_label, on = ['manufacturer', 'model', 'serial_number'], how = 'left')
        df_positive = df[~df['fault_time'].isna()]
        df = df[df['fault_time'].isna()]
        df.sample(n = len(df_positive) * 10, random_state = 2020, replace = True)
        df = pd.concat([df_positive, df])
        df.to_csv(data_path + data_pre_name + str(day) + '_sample_pn_v2.csv', index = False)
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
    df_train['dt'] = df_train['dt'].apply(lambda x: time.strftime('%Y-%m-%d', time.strptime(str(x), '%Y%m%d')))
    df_train['fault_time'] = df_train['fault_time'].fillna('1997-01-02')
    df_train['dt_day'] = df_train['dt'].apply(lambda x: int(time.mktime(time.strptime(str(x), "%Y-%m-%d")) / 24 / 3600))
    df_train['broken_day'] = df_train['fault_time'].apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d")) / 24 / 3600))
    df_train['dis_day'] = (df_train['broken_day'] - df_train['dt_day']).apply(lambda x: -1 if x < 0 else -1 if x >= 30 else x)
    # print df_train[df_train['dt'] == 20170709][['dt', 'fault_time', 'broken_day', 'dt_day', 'dis_day']].head(10)
    df_train['label'] = df_train['dis_day'].apply(lambda x: 1 if x != -1 else 0)
    weight = df_train['dis_day'].apply(lambda x: 1 if x == -1 else 30 - x)
    label = df_train['label']
    write_data_csv('df_train', df_train)
    write_data_csv('df_test', df_test)
    return df_train, df_test, weight, label

def get_weight_label_data(read_rows = int(sys.argv[1])):
    df_train = read_data_csv('df_train', read_rows)
    df_test = read_data_csv('df_test', read_rows)
    weight = df_train['dis_day'].apply(lambda x: 1 if x == -1 else 30 - x)
    label = df_train['label']
    return df_train, df_test, weight, label

def data_process(df_train, df_test, cols, col_ratio = 0.8):
    print '\ndata_process', '-' * 100
    df = pd.concat([df_train, df_test])
    drop_cols = []
    for col in cols:
        tmp = df[col].isna().sum() * 1.0 / len(df)
        print '%s\t%.5f' % (col, tmp)
        if tmp > col_ratio or col[-3 : ] == 'raw':
            drop_cols.append(col)
    print 'drop_col_ratio : %.3f, num of drop cols : %d last cols : %d' % (col_ratio, len(drop_cols), len(cols) - len(drop_cols))
    df = df.drop(drop_cols, axis = 1)
    for col in drop_cols:
        cols.remove(col)
    for col in cols:
        df[col] = df[col].astype(np.float64)
        df[col] = df[col].fillna(df[col].mean())
    stand = StandardScaler() # normalize
    datas = df[cols].values
    datas = stand.fit_transform(datas)
    df_tmp = pd.DataFrame(datas, columns = cols)
    df = pd.concat([df.drop(cols, axis = 1).reset_index(drop = True), df_tmp], axis = 1)

    df_train = df.iloc[: len(df_train), :]   
    df_test = df.iloc[len(df_train) : , :]   
    print cols
    return df_train, df_test, cols

def train(df_train, weight, label, df_test, cols, best_params):
    print '\ntrain' + '-' * 100
    clf = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=600, **best_params)
    clf.fit(X = df_train[cols], y = label, sample_weight = weight)
    pred = clf.predict(df_test[cols])
    df_test['pred'] = pred
    print 'test Counter :', Counter(df_test['pred'].tolist())
    df_test = df_test[df_test['pred'] == 1]
    return df_test

def select_fea(df_train, weight, label, cols, df_test = 0):
    params, best_score = val_lgb(df_train, weight, label, cols)
    df_result = train(df_train, weight, label, df_test, cols, params)
    df_result[['manufacturer', 'model', 'serial_number', 'dt']].to_csv(data_path + 'sub_20200301_1.csv', index = False, header = False)
    # best_score = 10
    # params = {}
    pre_cols = [x for x in cols]
    for col in cols:
        print '\ncol : %s\n', col
        temp_cols = [x for x in pre_cols]
        temp_cols.remove(col)
        params, score = val_lgb(df_train, weight, label, temp_cols)
        if score < best_score:
            best_score = score
            pre_cols.remove(col)
            df_result = train(df_train, weight, label, df_test, pre_cols, params)
            df_result[['manufacturer', 'model', 'serial_number', 'dt']].to_csv(data_path + 'sub_20200301_1.csv', index = False, header = False)
        break
    params, best_score = val_lgb(df_train, weight, label, cols)
    print 'cols change : ', len(cols), len(pre_cols)
    print 'score :', best_score
    return pre_cols, params

def main():
    pre_time = time.time()
    take_sample()
    df_train, df_test = read_data()
    print 'df_train.shape, df_test.shap : ', df_train.shape, df_test.shape
    # df_train, df_test, weight, label = get_weight_label_data()
    cols = df_test.columns.tolist()
    key_cols = ['manufacturer', 'model', 'serial_number', 'dt']
    for col in key_cols:
        if col in cols: cols.remove(col)
    df_train, df_test, weight, label = get_weight_label(df_train, df_test)
    print 'sum of weight : %d' % np.sum(weight)
    print 'sepend time : ', time.time() - pre_time
    print 'label.unique', Counter(label.tolist())

    df_train = df_train[key_cols + cols]
    df_test = df_test[key_cols + cols]
    df_train, df_test, cols = data_process(df_train, df_test, cols, col_ratio = 0.9)
    # df_result = train(df_train, df_test, cols, weight, label)
    cols, best_params = select_fea(df_train, weight, label, cols, df_test)

    write_data_csv('df_train_result', df_train)
    write_data_csv('df_test_result', df_test)
    print 'result cols : ', cols

    df_result = train(df_train, weight, label, df_test, cols, best_params)
    df_result[['manufacturer', 'model', 'serial_number', 'dt']].to_csv(data_path + 'sub_20200301_2.csv', index = False, header = False)
    pass
    print 'sepend time : ', time.time() - pre_time

main()
