# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:32:14 2023

@author: Zhengwu Long <longzhengwu2236@gmail.com>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score,balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
import shap
import scipy as sp
import scipy.cluster
import xgboost as xgb
import os
import catboost

#local file
from LIHC_Data_Prepare import load_TCGA_GEO_Data, load_real, get_fea_idx, rmbat_TCGA_GEO_data
from Pegasos_SVM import PegasosSVC_clip
from PSVC_Metric import Sensitivity_score, Specificity_score

def plot_shap(func, sv, figname=None, **args):
    savefig = True if figname is not None else False 
    func(sv, show=(not savefig),**args)
    
    if savefig:
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.show()
        
def plot_shap_dep(fea, sva, X, figname=None, **args):
    savefig = True if figname is not None else False 
    shap.dependence_plot(str(fea), sva, X, show=(not savefig), **args) # interaction_index=''
    
    if savefig:
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.show()

def plot_shap_allfig(model, X, datas, cols, max_display=15, labs=None, savefig=False, 
                     figpre="./fig_shap", figtype=".pdf", model_output="raw", dec_range = range(10)):
        
    explainer = shap.Explainer(model)
    
    
    if not os.path.exists(figpre):
        os.makedirs(figpre)
        
    sv_lst = {i:explainer(pd.DataFrame(datas[i], columns=cols), j) for i,j in datas}
    ev= explainer.expected_value
    sva_lst = {i:explainer.shap_values(datas[i]) for i in datas}
    
    cols_ = [i[4:] for i in cols]
    sv_lst_ = {i:explainer(pd.DataFrame(datas[i], columns=cols_)) for i in datas}
    
    
    for k in datas:
        shap_values = sv_lst[k]
        shap_values_a = sva_lst[k]
        
        # 微观图
        fname = os.path.join(figpre, "waterfall_"+k+ figtype) if savefig else None
        plot_shap(shap.plots.waterfall, shap_values[0], figname=fname, max_display=max_display)

        
        if savefig:
            fname = os.path.join(figpre, "force_"+ k + ".html")
            p = shap.force_plot(shap_values[0])
            shap.save_html(fname, p)
        
        # 宏观图
        if savefig:
            fname = os.path.join(figpre, "force_all_"+ k + ".html")
            p = shap.plots.force(shap_values)
            shap.save_html(fname, p)
        
        fname = os.path.join(figpre, "scatter_"+k+ figtype) if savefig else None
        # plot_shap(shap.plots.scatter, sv_lst_[k], figname=fname,
        #           ylabel="SHAP value\n(higher means more likely to renew)")
        plot_shap(shap.plots.scatter, shap_values, figname=fname,
                  ylabel="SHAP value\n(higher means more likely to renew)")
        
        fname = os.path.join(figpre, "bar_"+k+ figtype) if savefig else None
        plot_shap(shap.plots.bar, shap_values, figname=fname, max_display=max_display)
        
        fname = os.path.join(figpre, "sumplot_"+k+ figtype) if savefig else None
        plot_shap(shap.summary_plot, shap_values, figname=fname, max_display=max_display)

        fname = os.path.join(figpre, "heatmap_"+k+ figtype) if savefig else None
        plot_shap(shap.plots.heatmap, shap_values, figname=fname, max_display=max_display)

        fname = os.path.join(figpre, "decision_"+k+ figtype) if savefig else None
        expected_value = explainer.expected_value
        features_display = X.columns
        shap.decision_plot(expected_value, shap_values_a[dec_range], features_display, 
                           show=(not savefig), alpha=0.7, xlim=[-0.1,1.1])
        if savefig:
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.show()
        
        for mi in cols:
            fname = os.path.join(figpre, "dependence_"+mi + "_"+ k+ figtype) if savefig else None
            # print(shap_values_a.shape)
            plot_shap_dep(mi, shap_values_a, pd.DataFrame(datas[k], columns=cols), figname=fname)
            

def main():
    mirs = np.sort( pd.read_csv("cache/candi_feas.csv", index_col=0).iloc[:, 0])
    X_train, X_test, y_train, y_test, X_geo, y_geo, \
        X_TCGA, y_TCGA, miRNA_ID, gdata_sizes, clin_TCGA \
            = load_TCGA_GEO_Data(mirs=mirs,
                          over_sampling=False, 
                          mirs_del=[],trans_log=True,geomask=[3,4,7,8,9,11])
    X_TCGA_rmbat, X_geo_rmbat = rmbat_TCGA_GEO_data(X_TCGA, X_geo, gdata_sizes, method='harmony')
    X_all = np.vstack((X_TCGA_rmbat, X_geo_rmbat))
    # X_all = np.vstack((X_TCGA, X_geo))
    y_all = np.hstack((y_TCGA, y_geo))
    