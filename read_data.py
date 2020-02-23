#coding=utf-8
import numpy as np
import pandas as pd
import random
import time
# machine learning
# import xgboost
import warnings
warnings.filterwarnings("ignore")

data_path = r'D:\kaggle\data\tianchi_disk\a'[: -1]
# print '中文'
def change_data(data_name):
    now = time.time()
    df = pd.read_csv(data_path + data_name + '.csv')
    df.to_pickle(data_path + data_name + '.pkl', protocol = 2)

def read_data_pkl(data_name):
    df = pd.read_pickle(data_path + data_name + '.pkl')
    return df

def read_data_csv(data_name, mark = 100):
    if mark < 0:
        df = pd.read_csv(data_path + data_name + '.csv')
    else:
        df = pd.read_csv(data_path + data_name + '.csv', nrows = mark)
    return df

def check_data():
    begin_time = time.time()
    change_data('disk_sample_smart_log_201803')
    print 'change spend time :', time.time() - begin_time

    begin_time = time.time()
    df = read_data_csv('disk_sample_smart_log_201803', -1)
    print df.shape
    print 'csv spend time :', time.time() - begin_time

    begin_time = time.time()
    df = read_data_pkl('disk_sample_smart_log_201803')
    print df.shape
    print 'pkl spend time :', time.time() - begin_time
    pass

def main():
    check_data()

if __name__ == '__main__':
    main()
