# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:02:35 2023

@author: longz
"""

import numpy as np
import pandas as pd
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score, balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, make_scorer)
from sklearn.model_selection import StratifiedKFold, cross_validate

# import warnings
from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.simplefilter("ignore", category=ConvergenceWarning)

def Sensitivity_score(y_true, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred, labels=[1, 0]).T.ravel()
    if (TP == 0):
        return 0.
    else:    
        return TP / (TP + FN)
def Specificity_score (y_true, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred, labels=[1, 0]).T.ravel()
    if (TN == 0):
        return 0.
    else:
        return TN / (FP + TN)

clfs = {
        "DT": DecisionTreeClassifier(max_depth=6),
        "RF": RandomForestClassifier(n_estimators=100, max_depth=6),
        "AB": AdaBoostClassifier(n_estimators=100),
        "KNN": KNeighborsClassifier(),
        "LSVM": LinearSVC(C=0.001),
        # "RBF-SVM": SVC(C=0.07,gamma=0.08),
        "RBF-SVM": SVC(C=0.12, gamma=0.07),
        "NB": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(16, 8)),
        "XGB": XGBClassifier(max_depth=6, num_leaves=8, n_estimators=100, verbosity=0),
        "LGB": LGBMClassifier(n_estimators=100, verbosity=-1, boosting_type="goss", 
            max_depth=4, objective="binary", num_leaves=8, subsample=0.7,colsample_bytree=1.0),
        "CB": CatBoostClassifier(n_estimators=100, logging_level='Silent',allow_writing_files=False),
        }
#"XGB": XGBClassifier(max_depth=3, n_estimators=100, verbosity=0, min_child_weight=0.2,
# max_leaves=4, gamma=0.5, subsample=0.7, colsample_bytree=0.8), 
# "LGB": LGBMClassifier(n_estimators=100, verbosity=-1, boosting_type="gbdt", reg_lambda=0.1, 
# max_depth=3, objective="binary", num_leaves=8, subsample=0.7, colsample_bytree=0.9)

scorings=("matthews_corrcoef",'f1', 'balanced_accuracy')

def calc_clf_reward(X, y, support, clf, flen, flen_max, alpha=0.8, scoring="matthews_corrcoef", n_splits=5,
                    shuf=True, n_times=1, makeinfo=True, scocut=0.5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuf)
    scores = np.zeros((n_times, n_splits))
    for i in range(n_times):
        clf_imp = clfs[clf]
        
        scores_one = cross_validate(clf_imp, X[:, support], y, cv=cv,scoring=scoring, return_train_score=False, n_jobs=-1)
        scores[i, :]  = scores_one["test_score"]
    met_score =  np.median(scores)
    # flen_score =   (1 - flen/flen_max)**0.75
    flen_score = flen_max / flen if flen > flen_max else 1.
    info = ""
    if makeinfo:
        stdscore = np.std(scores)
        info = "Score: {:.3f}±{:.3f} fea:{}/{}".format(met_score, stdscore, flen, flen_max)
    # return (alpha * max(met_score - scocut, 0) / (1 - scocut) + (1 - alpha) * flen_score), info
    # return (alpha * met_score + (1 - alpha) * flen_score), info
    return max(0, met_score - scocut) * (flen_score), info

def print_clf_feas_met(support, X, y, clf, n_times=10, n_splits=10, shuf=True, params=None):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuf)
    scorings = {
                "ACC": "accuracy",
                "bACC": "balanced_accuracy",
                "f1": "f1",
                "MCC": "matthews_corrcoef",
                "Sensitivity": make_scorer(Sensitivity_score),
                "Specificity": make_scorer(Specificity_score),
                "AUC": "roc_auc"
        }
       
    clf_imp = copy.deepcopy(clfs[clf])
    
    if params is not None:
        clf_imp.set_params(**params)
    
    lst = []
    for i in range(n_times):
        scores_one = cross_validate(clf_imp, X[:, support], y, cv=cv,scoring=scorings, return_train_score=False, n_jobs=-1)
        lst.append(pd.DataFrame(scores_one))
    
    rst = pd.concat(lst, axis=0)
    rst = rst.reset_index()
    
    met_mean, met_std, met_med = {},{}, {}
    for k in scorings:
        k_ = "test_" + k
        met_mean[k] = rst[k_].mean()
        met_std[k] = rst[k_].std()
        met_med[k] = np.median(rst[k_])
    
    return pd.DataFrame(met_mean.values(), index=met_mean.keys(), columns=[clf]),\
           pd.DataFrame(met_std.values(), index=met_mean.keys(), columns=[clf]),\
           pd.DataFrame(met_med.values(), index=met_mean.keys(), columns=[clf])
    
    
def results_printer_v0(rst_dct, X, y, miRNA_ID, n_times=100):
    mean_lst = []
    med_lst = []
    infotmp = []
    for k in rst_dct.keys():
        mets = print_clf_feas_met(rst_dct[k].support_, X, y, k, n_times=n_times)
        mean_lst.append(mets[0].T)
        med_lst.append(mets[2].T)
        infotmp.append((k, miRNA_ID[rst_dct[k].support_]))

    mean_df = pd.concat(mean_lst)
    med_df = pd.concat(med_lst)
    
    for i in infotmp:
        # print(i[0], end=" ")
        for j in i[1]:
            print(j, end=" ")
        print()
    return mean_df, med_df
    
    
    


