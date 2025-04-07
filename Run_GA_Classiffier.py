# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:34:22 2023

@author: Zhengwu Long <longzhengwu2236@gmail.com>
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score,
                             precision_recall_fscore_support, roc_auc_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import ADASYN, BorderlineSMOTE

#local file
from Pegasos_SVM import PegasosSVC_clip
from LIHC_miRNA_Pannel_Optimization import print_fs_met
from Classifier_Reward import calc_clf_reward,print_clf_feas_met, clfs, results_printer_v0
from LIHC_Data_Prepare import load_TCGA_GEO_Data, load_real, tcga_mir_low_fliter, rmbat_TCGA_GEO_data

from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')


#遗传算法
class GAFeatureSelection():
    #K目标特征数, elite_num精英数量, clen编码长度
    #N染色体数，Tmax进化代数，pc交叉概率，pm变异概率
    def __init__(self, K=10, N=50, elite_num=2, clen=None, Tnml= None, Tlo = None,
                 Tmax=100, Pc=0.65, Pm=0.05,eps=1e-4,base=2):
        # if argNum != len(argRange):
        #     raise ValueError("argNum != len(argRange)")
        self.elite_num = elite_num // 2 * 2
        self.N,self.Tmax = N, Tmax
        self.Pc, self.Pm = Pc, Pm
        self.clen = clen
        self.codeBuffer = {}
        self.csets = None
        self.Tnml = Tnml if Tnml is not None else self.Tmax
        self.Tlo = Tlo if Tlo is not None else self.Tmax
        self.reward_store = None
        self.info_store = None        
        self.support_ = None
        self.K = K
        
    # @staticmethod
    # def gene2str(code, clen, temp="{:03d}"):
    #     # codeTmp = sorted(code.copy())
    #     strTmep = temp * clen
    #     return strTmep.format(*code)
    @staticmethod
    def gene2str(code, clen, temp="{:01d}"):
        # codeTmp = sorted(code.copy())
        strTmep = temp * clen
        return strTmep.format(*code)
    @staticmethod
    def str2gene(gene):
        return np.array(list(gene), dtype=int).astype(bool)
    
    def get_reward(self, X, y, support_, clf, scoring, makeinfo=True, n_times=1, n_splits=5):            
        if (np.sum(support_) == 0):
            return 0, "No select Feas!"

        flen = np.sum(support_)
        met, info = calc_clf_reward(X, y, support_, clf, flen, self.K,scoring=scoring,
                                    n_times=n_times,makeinfo=makeinfo, n_splits=n_splits, alpha=0.75)   

        return met, info
         
    def calc_fintness_genes(self, X, y, genes, clf, scoring, n_times, n_splits, usebuffer=False):
        if genes.ndim != 2:
            genes = genes.reshape((1, -1))
        stGenes = genes.copy()
        # stGenes.sort(axis=1)
        # print(genes[0], stGenes[0])
        fit_val = np.zeros((stGenes.shape[0]))
        infos = []
        for i in range(fit_val.shape[0]):
            strgene = self.gene2str(stGenes[i], self.clen, temp="{:1d}")
            if usebuffer and strgene in self.codeBuffer:
                fit_val[i], info  = self.codeBuffer[strgene]
                infos.append(info)
            else:
                met, info = self.get_reward(X, y, stGenes[i], clf, scoring,n_times, n_splits)
                # print(met, info)
                fit_val[i] = met
                infos.append(info)
                if usebuffer: self.codeBuffer[strgene] = [met, info]

        fit_max, fit_mean = fit_val.max(),fit_val.mean()
        fit_argidx = fit_val.argsort()
        elt_idx = fit_argidx[:-self.elite_num-1:-1]
        
        info = infos[fit_argidx[-1]]
        # print(elt_idx, self.elite_num)

        # elt_genes = genes[elt_idx]
        # if np.any(np.isnan(fit_val.sum())):
        #    print(fit_val)

        fit_cum = (fit_val / fit_val.sum()).cumsum()
        # print(fit_cum)
        return fit_cum, fit_max, fit_mean, elt_idx, info
    
    @staticmethod
    def selcct_cum(fit_cum):
        p = np.random.rand()
        idx = 0
        while (p > fit_cum[idx]):
            idx += 1
        return idx
       
    def selection(self, fit_cum):
        slc = np.zeros((2))
        slc[0] = self.selcct_cum(fit_cum)
        slc[1] = self.selcct_cum(fit_cum)
        while (slc[0] == slc[1]):
            slc[1] = self.selcct_cum(fit_cum)
        return slc.astype(int)
    
    def cross(self, slc, genes):
        p = np.random.rand()
        crsGene = genes[slc]
        
        if (p < self.Pc):
            pos = np.random.randint(self.clen)
            crsGene[0] = np.hstack((genes[slc[0],:pos], genes[slc[1],pos:]))
            crsGene[1] = np.hstack((genes[slc[1],:pos], genes[slc[0],pos:]))         
        return crsGene

    def mutate(self, crsGene):
        for i in range(crsGene.shape[0]):
            p = np.random.rand()
            if (p < self.Pm):
                pos = np.random.randint(self.clen)
                crsGene[i,pos] = 1 - crsGene[i,pos]
        return crsGene      
    
    def local_opt(self, genes, part=1.):
        n,c = genes.shape
        for i in range(n):
            if (np.random.rand() >= part):
                continue
            pos = np.random.randint(c)
            mutVal = (genes[i, pos] + 1) % len(self.csets)
            while((mutVal == genes[i,:]).any()):
                mutVal = (mutVal + 1) % len(self.csets)
            genes[i, pos] = mutVal
        return genes
    
    
    
    def plot_ga_log(self, col_mask=[], rg=None):
        df = self.Sarsa_get_log(col_mask)
        x = df.index
        if rg is None:
            idx = range(0, df.shape[0])
        else:
            idx = range(rg[0], rg[1])
    
        plt.figure(dpi=600)
        for i in df.columns:
            plt.plot(x[idx], df.loc[idx, i], label=i)
        plt.legend()
        plt.show()
    
    def get_bestInfo(self, allfeas):
        maxit = np.argmax(self.reward_store)
        return allfeas[self.support_], self.info_store[maxit][1]
    
    def init_genes(self):
        return np.random.binomial(1, 0.4, size=((self.N, self.clen))).astype(bool)
        
    def calc(self, X, y, clf="LGB", scoring="matthews_corrcoef", n_times=5, n_splits=10, prt=True):
        clen = X.shape[1]
        self.clen = clen
        
        genes = self.init_genes()
        # print(genes.sum(axis=1), genes.shape)
        
        reward_store = np.zeros((self.Tmax))
        info_store = []
        
        fit_cum, fit_max, fit_mean, elt_idx, info =  self.calc_fintness_genes(X, y, genes, clf, scoring, n_times, n_splits)
        
        # bst_max, bst_mean, bst_gene = [], [], []
        # bst_max.append(fit_max); bst_mean.append(fit_mean)
        # bst_gene.append(genes[elt_idx[0]])
        
        for j in tqdm(range(self.Tmax)): #T times iteration
            # print("GA_main processing: {:2d}/{:2d}".format(i, self.Tmax))
            newGenes = np.zeros_like(genes)
            for i in range(0, self.N - self.elite_num, 2): #selection queue using fitness value
                slc = self.selection(fit_cum);
                tmpGene = self.cross(slc, genes)
                tmpGene = self.mutate(tmpGene)
                newGenes[i:i+2, :] = tmpGene
            newGenes[-self.elite_num:] = genes[elt_idx] #elite enter queue
            
            #recalculate newGenes fitness
            genes = newGenes

            fit_cum, fit_max, fit_mean, elt_idx, info = self.calc_fintness_genes(X, y, genes, clf, scoring, n_times, n_splits)
            reward_store[j] = fit_max
            info_store.append((self.gene2str(genes[elt_idx[0]], clen), info))
            
            # bst_max.append(fit_max); bst_mean.append(fit_mean)
            # bst_gene.append(genes[elt_idx[0]])
            if prt:
                print("reward:",fit_max, "info", info)
        
        max_fit_idx = np.argmax(reward_store)
        selected_features = info_store[max_fit_idx][0]

        # save 
        self.support_ = self.str2gene(selected_features).astype(bool)
        self.info_store = info_store
        self.reward_store = reward_store
        
        
        return self

mirs = np.sort( pd.read_csv("cache/candi_feas.csv", index_col=0).iloc[:, 0])
X_train, X_test, y_train, y_test, X_geo, y_geo, \
    X_TCGA, y_TCGA, miRNA_ID, gdata_sizes, clin_TCGA \
        = load_TCGA_GEO_Data(mirs=mirs,
                      over_sampling=False, 
                      mirs_del=[],trans_log=True,geomask=[3, 4, 7, 8, 9, 11]      )
                      # mirs_del=[],trans_log=True,geomask=[2, 3, 7, 8, 9, 11])

# X_all = np.vstack((X_TCGA, X_geo))
X_TCGA_rmbat, X_geo_rmbat = rmbat_TCGA_GEO_data(X_TCGA, X_geo, gdata_sizes, method='harmony')
# X_TCGA_rmbat, X_geo_rmbat = X_TCGA, X_geo   
X_all = np.vstack((X_TCGA_rmbat, X_geo_rmbat))
y_all = np.hstack((y_TCGA, y_geo))

from Classifier_Reward import clfs, print_clf_feas_met
rst_dct = {}
# ga_fe = GAFeatureSelection(K=10, Tmax=1)
# ga_fe.calc(X_all, y_all, clf="MLP", n_times=5, n_splits=10)  

for k in clfs:
    # if k not in {"LGB", "XGB", "RF", "MLP", "KNN", "RBF-SVM"}: continue 
    # if k not in {"LGB", "XGB", "RBF-SVM"}: continue
    # if k not in {"CB"}:  continue    
    if k not in {'KNN', 'LSVM', 'RBF-SVM'}:  continue
    print(k, "start!")
    ga_fe = GAFeatureSelection(K=10, Tmax=30)
    ga_fe.calc(X_all, y_all, clf=k, n_times=5, n_splits=10)
    rst_dct[k] = ga_fe
    print(ga_fe.get_bestInfo(miRNA_ID))
    print(print_clf_feas_met(ga_fe.support_, X_all, y_all, k)[0].T)


mean_df, med_df = results_printer_v0(rst_dct, X_all, y_all, miRNA_ID)
    
