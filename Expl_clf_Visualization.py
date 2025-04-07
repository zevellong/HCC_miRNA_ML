# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:42:33 2023

@author: Zhengwu Long <longzhengwu2236@gmail.com>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import pickle
import copy
from sklearn.inspection import permutation_importance
from pdpbox import pdp, info_plots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP #pip install umap-learn
import catboost
from alibi.explainers import ALE, plot_ale
from alepython import ale_plot
from xgboost import XGBClassifier

#local file
from Pegasos_SVM import PegasosSVC_clip
from LIHC_miRNA_Pannel_Optimization import print_fs_met
from Classifier_Reward import calc_clf_reward,print_clf_feas_met, clfs
from LIHC_Data_Prepare import load_TCGA_GEO_Data, load_real, tcga_mir_low_fliter,get_fea_idx, rmbat_TCGA_GEO_data

def mir_rename(mirs):
    return [i.replace("hsa-mir-", "miR-") for i in mirs]


######## 读入数据 ############
mirs = np.sort( pd.read_csv("cache/candi_feas.csv", index_col=0).iloc[:, 0])
X_train, X_test, y_train, y_test, X_geo, y_geo, \
    X_TCGA, y_TCGA, miRNA_ID, gdata_sizes, clin_TCGA \
        = load_TCGA_GEO_Data(mirs=mirs,
                      over_sampling=False, 
                      mirs_del=[],trans_log=True,geomask=[3, 4, 7, 8, 9, 11])
                      # mirs_del=[],trans_log=True,geomask=[2, 3, 7, 8, 9, 11])

mirs = "hsa-mir-101-1 hsa-mir-122 hsa-mir-1269a hsa-mir-139 hsa-mir-146a hsa-mir-183 hsa-mir-224 hsa-mir-483 hsa-mir-589 hsa-mir-93".split(" ")
idx = get_fea_idx(mirs, miRNA_ID)
# X_all = np.vstack((X_TCGA, X_geo))
y_all = np.hstack((y_TCGA, y_geo))

X_TCGA_rmbat, X_geo_rmbat = rmbat_TCGA_GEO_data(X_TCGA, X_geo, gdata_sizes, method='harmony')
X_all = np.vstack((X_TCGA_rmbat, X_geo_rmbat))


params = {'l2_leaf_reg': 0.1, 'loss_function': 'Logloss', 'max_depth': 4, 'n_estimators': 250}
clf = catboost.CatBoostClassifier(n_estimators=100, logging_level='Silent',
                                  allow_writing_files=False)
clf = XGBClassifier()
clf.set_params(**params)
clf.fit(X_all[:, idx], y_all)


import plotly.io as pio
pio.renderers.default = "pdf"
import os
import sklearn
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import ALE

feature_names = mirs + ["target"]
data = X_all[:, idx]
data = pd.DataFrame(np.hstack([X_all[:, idx], y_all.reshape([y_all.shape[0], -1])]), 
                    columns=feature_names)
tabular_data = Tabular(
    data,
    feature_columns=feature_names,
    categorical_columns=None,
    target_column="target",
)

transformer = TabularTransform().fit(tabular_data)
class_names = transformer.class_names
x = transformer.transform(tabular_data)
train, test, train_labels, test_labels = \
    sklearn.model_selection.train_test_split(x[:, :-1], x[:, -1], train_size=0.80)

predict_function=lambda z: clf.predict_proba(transformer.transform(z))

explainer = ALE(
    training_data=tabular_data,
    predict_function=predict_function
)

explanations = explainer.explain()
explanations.ipython_plot(class_names=class_names)
fig, axs = plt.subplots(5, 2)
mtt = explanations.plot(axs)
dtt = explanations.plotly_plot(ncols=5)


from omnixai.data.tabular import Tabular
from omnixai.preprocessing.base import Identity
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import SensitivityAnalysisTabular

explainer = SensitivityAnalysisTabular(
    training_data=tabular_data,
    predict_function=predict_function,
)
explanations = explainer.explain()
explanations.ipython_plot()

from interpret import show


from interpret import set_visualize_provider
from interpret.provider import InlineProvider
# set_visualize_provider(InlineProvider())
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from interpret import show
from interpret.blackbox import MorrisSensitivity
seed = 42
np.random.seed(seed)
X, y = data.iloc[:,:-1], data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

pca = PCA()
rf = RandomForestClassifier(random_state=seed)

blackbox_model = Pipeline([('cbm', clf)])
blackbox_model.fit(X_train, y_train)

msa = MorrisSensitivity(blackbox_model, X_train)
msa.explain_global()
msa1 = msa.explain_global()


predict_function = lambda z: rf.predict(transformer.transform(z))
explainer = SensitivityAnalysisTabular(
    training_data=tabular_data,
    predict_function=predict_function,
)
explanations = explainer.explain()


import seaborn as sns
fig, axes = plt.subplots(2, 2)
datas = [[msa.mu_, np.asarray(msa.mu_star_)], [msa.sigma_, np.asarray(msa.mu_star_conf_)]]
df = pd.DataFrame([msa.mu_, np.asarray(msa.mu_star_), msa.sigma_,
                   np.asarray(msa.mu_star_conf_)], columns=mir_rename(mirs), 
                  index=["mu", "mu_star","sigma", "mu_star_conf"]).T


for i in range(2):
    for j in range(2):
        sns.barplot(data=df.iloc[:,i*2+j], orient = 'h',
                     ax=axes[i,j],).set(title=df.index[i*2+j]).set(yticklabels=[])

from omnixai.explainers.tabular import PartialDependenceTabular
explainer = PartialDependenceTabular(
    training_data=tabular_data,
    predict_function=predict_function
)

explanations = explainer.explain()
# explanations.ipython_plot(class_names=class_names)
explanations.plotly_plot(class_names=["Normal", "HCC"])

pdptt = explanations.plotly_plot(class_names=["Normal", "HCC"], ncols=5)
pdptt.component.update_layout(
    margin=dict(l=20, r=80, t=20, b=10),
    width=950, height=450,
)

from omnixai.explainers.tabular import PartialDependenceTabular
explainer = PartialDependenceTabular(
    training_data=tabular_data,
    predict_function=predict_function
)

explanations = explainer.explain()
# explanations.ipython_plot(class_names=class_names)
explanations.plotly_plot(class_names=["Normal", "HCC"], ncols=5)
