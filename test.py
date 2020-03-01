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
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
import xgboost
import datetime
# plot
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

# myself libs
# from read_data import read_data_csv, read_data_pkl
# other
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_path = r'D:\kaggle\data\tianchi_disk\a'[: -1]
pic_path = r'picture\a'[: -1]

print 'test.py'

def main():
    x = [1, 2, 23]
    y = [i for i in x]
    y.remove(1)
    print x
    print y

main()