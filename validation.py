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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, roc_auc_score, f1_score, make_scorer, precision_score, recall_score
from collections import Counter
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import xgboost
import lightgbm as lgb
import datetime
# myself libs
# other
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_path = r'D:\kaggle\data\tianchi_disk\a'[: -1]
print 'data_path :', data_path

cols, X_train, y_train, weight = 0, 0, 0, 0

def objective(params):
    time1 = time.time()
    params = {
        'max_depth': int(params['max_depth']),
        # 'gamma': "{:.3f}".format(params['gamma']),
        'subsample': "{:.2f}".format(params['subsample']),
        'reg_alpha': "{:.3f}".format(params['reg_alpha']),
        'reg_lambda': "{:.3f}".format(params['reg_lambda']),
        'learning_rate': "{:.3f}".format(params['learning_rate']),
        'num_leaves': '{:.0f}'.format(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'min_child_samples': '{:.0f}'.format(params['min_child_samples']),
        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])
    }
    # print("\n############## New Run ################")
    # print 'params = ', params
    FOLDS = 5
    tss = TimeSeriesSplit(n_splits=FOLDS)
    score_mean = 0
    for tr_idx, val_idx in tss.split(X_train, y_train):
        clf = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=600, **params)
        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        weight_tr, weight_vl = weight.iloc[tr_idx], weight.iloc[val_idx]
        clf.fit(X_tr, y_tr, sample_weight = weight_tr)
        pred = clf.predict(X_vl)
        score = f1_score(y_pred = pred, y_true = y_vl)
        score_mean += score
    # print 'Mean f1_score : ', score_mean * 1.0 / FOLDS, '\n'
    del X_tr, X_vl, y_tr, y_vl, clf, score
    gc.collect()
    return -(score_mean / FOLDS)

def val_lgb(X, weight_param, y, cols = [], params={}):
    print 'val_lgb', '-' * 100
    global X_train, y_train, weight
    weight = weight_param
    X_train = X[cols].copy()
    y_train = y.copy()
    print 'shape : ', X_train.shape, weight.shape
    space = {
        'max_depth': hp.quniform('max_depth', 7, 23, 1),
        'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
        'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, .9),
        # 'gamma': hp.uniform('gamma', 0.01, .7),
        'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),
        'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),
        'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),
        'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.4, .9)
    }
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals = 10)
    best_params = space_eval(space, best)
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['min_child_samples'] = int(best_params['min_child_samples'])
    f1_score = objective(best_params)
    print 'best params : ', best_params
    print 'f1_score : ', f1_score, '\n'
    return best_params, f1_score


def val_kSplit(X_train, weight, y_train, cols):
    kf = KFold(n_splits=5, shuffle = True, random_state = 2020)
    f1_score_all = 0
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        clf = lgb.LGBMClassifier()
        clf.fit(X_tr[cols], y_tr)
        pred = clf.predict(X_vl[cols])
        score_precision = precision_score(y_true = y_vl, y_pred = pred)
        score_recall = recall_score(y_true = y_vl, y_pred = pred)
        score_f1 = f1_score(y_true = y_vl, y_pred = pred)
        f1_score_all += score_f1
        print 'precision : %.4f, recall : %.4f, f1_score : %.4f' % (score_precision, score_recall, score_f1)
    print ''
    return f1_score_all / 5

def val_TimeSeriesSplit(X_train, weight, y_train, cols):
    FOLDS = 5
    tss = TimeSeriesSplit(n_splits=FOLDS)
    score_all = 0
    for tr_idx, val_idx in tss.split(X_train, y_train):
        clf = lgb.LGBMClassifier()
        X_tr, X_vl = X_train[cols].iloc[tr_idx, :], X_train[cols].iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        weight_tr, weight_vl = weight.iloc[tr_idx], weight.iloc[val_idx]
        clf.fit(X_tr[cols], y_tr)
        pred = clf.predict(X_vl[cols])
        score_all += (f1_score(y_pred = pred, y_true = y_vl))
    # print 'Mean f1_score : ', score_mean * 1.0 / FOLDS, '\n'
    return score_all / 5

def val_kSplit_weight(X_train, weight, y_train, cols):
    kf = KFold(n_splits=5, shuffle = True, random_state = 2020)
    f1_score_all = 0
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        weight_tr, weight_vl = weight.iloc[tr_idx], weight.iloc[val_idx]
        clf = lgb.LGBMClassifier()
        clf.fit(X_tr[cols], y_tr, sample_weight = weight_tr)
        pred = clf.predict(X_vl[cols])
        score_precision = precision_score(y_true = y_vl, y_pred = pred)
        score_recall = recall_score(y_true = y_vl, y_pred = pred)
        score_f1 = f1_score(y_true = y_vl, y_pred = pred)
        f1_score_all += score_f1
        print 'precision : %.4f, recall : %.4f, f1_score : %.4f' % (score_precision, score_recall, score_f1)
    print ''
    return f1_score_all / 5

def val_TimeSeriesSplit_weight(X_train, weight, y_train, cols):
    FOLDS = 5
    tss = TimeSeriesSplit(n_splits=FOLDS)
    score_mean = 0
    for tr_idx, val_idx in tss.split(X_train, y_train):
        clf = lgb.LGBMClassifier()
        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        weight_tr, weight_vl = weight.iloc[tr_idx], weight.iloc[val_idx]
        clf.fit(X_tr[cols], y_tr, sample_weight = weight_tr)
        pred = clf.predict(X_vl[cols])
        score = f1_score(y_pred = pred, y_true = y_vl)
        score_mean += score
    return score_mean / FOLDS

