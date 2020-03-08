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
import matplotlib.pyplot as plt
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

def dropna_ratio(day):
    data_pre_name = 'disk_sample_smart_log_'
    data_name = data_pre_name + str(day) + '.csv'
    df = pd.read_csv(data_path + data_name).sample(frac = 0.5, random_state = 2020)
    cnt = len(df)
    x = df.isna()
    del df
    gc.collect()
    x = x.sum() * 1.0 / cnt
    plt.hist(x = x, bins = 20)
    plt.title(str(day) + "_" + str((x < 0.2).sum()))
    plt.savefig('./picture/' + str(day) + '.jpg')

    del x
    gc.collect()

def main():
    for day in range(201707, 201713) + range(201801, 201808):
        dropna_ratio(day)

main()