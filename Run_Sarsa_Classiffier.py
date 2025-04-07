# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:23:12 2023

@author: Zhengwu Long <longzhengwu2236@gmail.com>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score,
                             precision_recall_fscore_support, roc_auc_score, matthews_corrcoef)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tqdm import tqdm

#local file
from Pegasos_SVM import PegasosSVC_clip
from LIHC_miRNA_Pannel_Optimization import print_fs_met
from Classifier_Reward import calc_clf_reward,print_clf_feas_met,results_printer_v0
from LIHC_Data_Prepare import load_TCGA_GEO_Data, load_real, tcga_mir_low_fliter, rmbat_TCGA_GEO_data




class SarsaFeatureSelection:
    def __init__(self,  iterations=200, K=10, epsilon=0.5, alpha=0.2,
                 epsilon_decay_rate=0.995, alpha_decay_rate=0.995):
        self.K = K
        self.iterations = iterations
        self.epsilon = epsilon
        self.alpha = alpha
        self.epsilon_decay_rate = epsilon_decay_rate
        self.alpha_decay_rate = alpha_decay_rate
        self.n_features_in_ = 0  # 原始特征数
        self.actions = None
        self.support_ = None
        self.codeBuffer = {}
        self.hit_count = 0
        self.feature_store = None
        self.info_store = None
    @staticmethod
    def gene2str(code, clen, temp="{:01d}"):
        # codeTmp = sorted(code.copy())
        strTmep = temp * clen
        return strTmep.format(*code)
    @staticmethod
    def str2gene(gene):
        return np.array(list(gene), dtype=int).astype(bool)
    
    def get_reward(self, X, y, support_, clf, scoring, usecache=False, makeinfo=True, n_times=1, n_splits=5):            
        if (np.sum(support_) == 0):
            return 0, "No select Feas!"

        ## 训练分类器
        strfea = SarsaFeatureSelection.gene2str(support_, self.n_features_in_)
        if usecache and strfea in self.codeBuffer:
            met, info = self.codeBuffer[strfea]
            self.hit_count += 1 #缓存命中次数
        else:
            flen = np.sum(support_)
            met, info = calc_clf_reward(X, y, support_, clf, flen, self.K,scoring=scoring,
                                        n_times=n_times,makeinfo=makeinfo, n_splits=n_splits, alpha=0.75)   
            if usecache:
                self.codeBuffer[strfea] = [met, info]
        return met, info
    
    def get_cache_hit_rate(self):
        return self.hit_count / (self.hit_count + len(self.codeBuffer))
    
    
    
    def calc(self, X, y, clf, scoring="matthews_corrcoef", n_times=1, n_splits=5):
        """
        Iteratively selects features using SARSA algorithm.
    
        Returns:
        -------
        selected_features : array-like, shape (n_selected_features,)
            Indices of the selected features.
        """
        self.n_features_in_ = X.shape[1] # 原始特征数
        self.actions = np.zeros(self.n_features_in_, dtype=int) #智能体
        
        epsilon = self.epsilon
        alpha = self.alpha
        epsilon_decay_rate = self.epsilon_decay_rate
        alpha_decay_rate = self.alpha_decay_rate
        num_agents = self.n_features_in_
        
        Q_values = np.zeros((num_agents, 2))
        actions = np.zeros(num_agents, dtype=int)
    
        # Iteratively select features using SARSA algorithm
        reward_store = np.zeros(self.iterations)
        info_store = []
        
        # for it in (range(self.iterations)):
        for it in tqdm(range(self.iterations)):
            # Iterate over agents and set action for each agent
            for agent in range(num_agents):
                rand_number1 = np.random.random()
                rand_number2 = np.random.random()
                if rand_number1 > epsilon:
                    # Select action with maximum Q value
                    y_value = np.argmax(Q_values[agent, :])
                    actions[agent] = bool(y_value)
                else:
                    # Randomly select action
                    if rand_number2 >  0.6:#epsilon:
                        actions[agent] = 1
                    else:
                        actions[agent] = 0

            # Calculate reward for selected features
            features = np.where(actions == 1)[0]
            # print(features, X_train.shape)
            fit, info = self.get_reward(X, y, actions>0, clf, scoring=scoring, makeinfo=True, n_times=n_times, 
                                        n_splits=n_splits)
            strfeas = self.gene2str(actions, num_agents)
            if actions.sum() <= 5:
                print(f'Iteration: {it}  Fitness: {fit:.4f}  {info}  Feas: {features}')
            else:
                print(f'Iteration: {it}  Fitness: {fit:.4f}  {info}')
            bstfit, bstinfo = fit, (strfeas, info)
            # Update Q values for each agent
            for agent in range(num_agents):
                # Flip action for the current agent
                save_action = actions[agent]
                actions[agent] = 1 - save_action
                new_fit, new_info = self.get_reward(X, y, actions.astype(bool), clf, scoring=scoring, makeinfo=True, 
                                             n_times=int(n_times), n_splits=n_splits)
                if (new_fit > bstfit):
                    strfeas = self.gene2str(actions, num_agents)
                    bstfit, bstinfo = new_fit, (strfeas, new_info)  
                C_agent =  new_fit - fit
                Q_values[agent, actions[agent]] += alpha * (C_agent - Q_values[agent, actions[agent]])
                actions[agent] = save_action
                
 
            # Decay learning rate and exploration probability
            alpha *= alpha_decay_rate
            epsilon *= epsilon_decay_rate
    
            # Save results
            reward_store[it] = fit
            info_store.append( bstinfo ) 
        
        # Select features with maximum reward
        max_fit_idx = np.argmax(reward_store)
        selected_features = info_store[max_fit_idx][0]
        
        # save 
        self.support_ = self.str2gene(selected_features).astype(bool)
        self.info_store = info_store
        self.reward_store = reward_store
        # return selected_features
        return self
    
    def transform(self, X):
        return X[:, self.support_]

    def Sarsa_get_log(self, col_mask=[]):
        itmax = self.iterations
        cols = ["sco_mean", "sco_std", "feas_num", "reward", "max_sco", "max_reward"]
        df = pd.DataFrame(np.zeros((itmax, len(cols))), columns=cols)
        max_reward = -1.
        max_sco = -1.
        nfeas = self.n_features_in_
        
        for i in range(itmax):
            iterinfo = self.info_store[i]
            flen = iterinfo[0].count('1') / nfeas
            sco, sco_std = [float(i) for i in iterinfo[1].split(" ")[1].split("±")]
            reward = self.reward_store[i]
            max_reward = max(max_reward, reward)
            max_sco = max(max_sco, sco)
            
            df.iloc[i, :] = [sco, sco_std, flen, reward, max_sco, max_reward]
        
        for i in col_mask:
            cols.remove(i)
    
        return df.loc[:, cols]
    

    def plot_sarsa_log(self, col_mask=[], rg=None):
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
    
    def Sarsa_get_bestInfo(self, allfeas):
        maxit = np.argmax(self.reward_store)
        return allfeas[self.support_], self.info_store[maxit][1]
    

mirs = np.sort( pd.read_csv("cache/candi_feas.csv", index_col=0).iloc[:, 0])
X_train, X_test, y_train, y_test, X_geo, y_geo, \
    X_TCGA, y_TCGA, miRNA_ID, gdata_sizes, clin_TCGA \
        = load_TCGA_GEO_Data(mirs=mirs,
                      over_sampling=False, 
                      mirs_del=[],trans_log=True,geomask=[3, 4, 7, 8,9,11])

X_TCGA_rmbat, X_geo_rmbat = rmbat_TCGA_GEO_data(X_TCGA, X_geo, gdata_sizes, method="harmony")
# X_TCGA_rmbat, X_geo_rmbat = X_TCGA, X_geo
X_all = np.vstack((X_TCGA_rmbat, X_geo_rmbat))
y_all = np.hstack((y_TCGA, y_geo))

# Sarsa_fe = SarsaFeatureSelection(iterations=20)
# Sarsa_fe.calc(X_all, y_all, clf="LGB", n_times=1)
# Sarsa_fe.plot_sarsa_log(["max_sco", "max_reward", "sco_std"])
# print(Sarsa_fe.Sarsa_get_bestInfo(miRNA_ID))
# print(print_clf_feas_met(Sarsa_fe.support_, X_all, y_all, "LGB")[0].T)


from Classifier_Reward import clfs
rst_dct = {}
for k in clfs:
    # if k not in {"LGB"}:  continue
    if k not in {'KNN', 'LSVM', 'RBF-SVM'}:  continue
    print(k, "start!")
    Sarsa_fe = SarsaFeatureSelection(iterations=30, K=10)
    Sarsa_fe.calc(X_all, y_all, clf=k, n_times=10, n_splits=10)
    rst_dct[k] = Sarsa_fe
    Sarsa_fe.plot_sarsa_log(["max_sco", "max_reward", "sco_std"])
    print(Sarsa_fe.Sarsa_get_bestInfo(miRNA_ID))
    print(print_clf_feas_met(Sarsa_fe.support_, X_all, y_all, k)[0].T)

