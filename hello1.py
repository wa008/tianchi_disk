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
from validation import val_lgb, val_kSplit, val_TimeSeriesSplit, val_kSplit_weight, val_TimeSeriesSplit_weight
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
        # if day <= 201806: continue
        df = read_data_csv(data_pre_name + str(day), read_rows)
        print 'raw : ', len(df)
        cnt = df.shape
        for col in ['manufacturer', 'model', 'serial_number']: df[col] = df[col].astype(str)
        df = df.merge(df_label, on = ['manufacturer', 'model', 'serial_number'], how = 'left')
        df_positive = df[~df['fault_time'].isna()]
        df = df[df['fault_time'].isna()]
        df = df.sample(n = len(df_positive) * 10, random_state = 2020)
        df = pd.concat([df_positive, df])
        df.to_csv(data_path + data_pre_name + str(day) + '_sample_pn_v3.csv', index = False)
        print 'sample : ', day, cnt, len(df_positive), len(df)
        del df, df_positive
        gc.collect()
    # return df

def read_data():
    read_rows = int(sys.argv[1])
    data_pre_name = 'disk_sample_smart_log_'
    df = read_data_csv(data_pre_name + '201707' + '_sample_pn_v2', read_rows)
    print '201707', df.shape
    for day in range(201708, 201713) + range(201801, 201808):
        df_temp = read_data_csv(data_pre_name + str(day) + '_sample_pn_v2', read_rows)
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
    
    df_black = df_train[df_train['label'] == 1]
    df_white = df_train[df_train['label'] == 0]
    df_white = df_white.sample(n = min(len(df_white), 5 * len(df_black)), random_state = 2020)
    df_train = pd.concat([df_white, df_black])
    df_train = df_train.sort_values(by = 'dt')

    df_train['temp'] = df_train['dis_day'].apply(lambda x: 0 if x == -1 else 30 - x)
    white_weight = df_train['temp'].sum() * 1.0 / len(df_train[df_train['temp'] == 0])
    weight = df_train['temp'].apply(lambda x: white_weight if x == 0 else x)
    label = df_train['label']
    # write_data_csv('df_train_get_weight', df_train)
    # write_data_csv('df_test_get_weight', df_test)
    return df_train, df_test, weight, label

def get_weight_label_data(read_rows = int(sys.argv[1])):
    df_train = read_data_csv('df_train', read_rows)
    df_test = read_data_csv('df_test', read_rows)
    weight = df_train['dis_day'].apply(lambda x: 1 if x == -1 else 30 - x)
    label = df_train['label']
    return df_train, df_test, weight, label

def data_process(df_train, df_test, cols, col_ratio = 0.8):
    print '\ndata_process', '-' * 100
    drop_cols = []
    for col in cols:
        tmp = df_train[col].isna().sum() * 1.0 / len(df_train)
        if tmp > col_ratio or col[-5 : ] == 'lized':
            drop_cols.append(col)
    # print 'drop_col_ratio : %.3f, num of drop cols : %d last cols : %d' % (col_ratio, len(drop_cols), len(cols) - len(drop_cols))
    df_train = df_train.drop(drop_cols, axis = 1)
    df_test = df_test.drop(drop_cols, axis = 1)
    for col in drop_cols:
        cols.remove(col)
    for col in cols:
        df_train[col] = df_train[col].astype(np.float64).fillna(df_train[col].mean())
        df_test[col] = df_test[col].astype(np.float64).fillna(df_train[col].mean())
    stand = StandardScaler() # normalize
    df_tmp = pd.DataFrame(stand.fit_transform(df_train[cols]), columns = cols)
    df_train = pd.concat([df_train.drop(cols, axis = 1).reset_index(drop = True), df_tmp], axis = 1)

    df_tmp = pd.DataFrame(stand.transform(df_test[cols]), columns = cols)
    df_test = pd.concat([df_test.drop(cols, axis = 1).reset_index(drop = True), df_tmp], axis = 1)
    return df_train, df_test, cols

def train(df_train, weight, label, df_test, cols, is_weight, out_id):
    # print '\ntrain' + '-' * 100
    # clf = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=600, **best_params)
    clf = lgb.LGBMClassifier()
    if is_weight == True:
        clf.fit(X = df_train[cols], y = label , sample_weight = weight)
    else:
        clf.fit(X = df_train[cols], y = label)
    pred = clf.predict(df_test[cols])
    df_test['pred'] = pred
    df_test = df_test[df_test['pred'] == 1]
    df_test[['manufacturer', 'model', 'serial_number', 'dt']].to_csv(data_path + 'sub_20200308_' + str(out_id) + '.csv', index = False, header = False)
    return df_test

def select_fea(df_train, weight, label, cols, df_test = 0, is_weight = False, out_id = 3):
    best_score_kSplit = val_kSplit(df_train, weight, label, cols)
    best_score_kTime = val_TimeSeriesSplit(df_train, weight, label, cols)
    # df_result = train(df_train, weight, label, df_test, cols, is_weight = False, out_id = 3)
    # best_score = 10
    # params = {}
    pre_cols = [x for x in cols]
    for col in cols:
        temp_cols = [x for x in pre_cols]
        temp_cols.remove(col)
        if is_weight == True:
            score_kSplit = val_kSplit_weight(df_train, weight, label, temp_cols)
            score_kTime = val_TimeSeriesSplit_weight(df_train, weight, label, temp_cols)
        else:
            score_kSplit = val_kSplit(df_train, weight, label, temp_cols)
            score_kTime = val_TimeSeriesSplit(df_train, weight, label, temp_cols)
        print 'col : %s, score_kSplit : %.4f, score_kTime : %.4f\n' % (col, score_kSplit, score_kTime)
        if score_kSplit > best_score_kSplit:
            best_score_kSplit = score_kSplit
            best_score_kTime = score_kTime
            pre_cols.remove(col)
            # df_result = train(df_train, weight, label, df_test, temp_cols)
    print '\nbest score_kSplit : %.4f, score_kTime : %.4f\n' % (best_score_kSplit, best_score_kTime)
    train(df_train, weight, label, df_test, cols, is_weight, out_id)
    print 'pre_cols : ', pre_cols
    print 'len(cols) : ', len(pre_cols)
    return pre_cols

def main():
    pre_time = time.time()
    # take_sample()
    df_train, df_test = read_data()

    print 'df_train.shape, df_test.shap : ', df_train.shape, df_test.shape
    # df_train, df_test, weight, label = get_weight_label_data()
    cols = df_test.columns.tolist()
    key_cols = ['manufacturer', 'model', 'serial_number', 'dt']
    for col in key_cols:
        if col in cols: cols.remove(col)
    df_train, df_test, weight, label = get_weight_label(df_train, df_test)
    print 'df_train.shape, df_test.shap : ', df_train.shape, df_test.shape
    print 'weight of black %d, white : %d' % (np.sum(weight[df_train['label'] == 1]), np.sum(weight[df_train['label'] == 0]))
    print 'label.unique', Counter(label.tolist())
    print 'sepend time : ', time.time() - pre_time

    df_train, df_test, cols = data_process(df_train, df_test, cols, col_ratio = 0.9)

    print 'result cols : ', cols
    print 'len(cols) : ', len(cols)

    print 'has noweith' + '-' * 50
    cols = select_fea(df_train, weight, label, cols, df_test, False, 4)
    print 'has weith' + '-' * 50
    cols = select_fea(df_train, weight, label, cols, df_test, True, 5)

    print 'sepend time : ', time.time() - pre_time

main()
# main_middle()
