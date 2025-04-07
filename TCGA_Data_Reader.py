# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:26:22 2022

Read in the original TCGA miRNA expression file and clinical information, 
delete invalid rows and columns, save as hdf5 file
读取原始TCGA miRNA表达文件和临床信息，删除无效的行和列，另存为 hdf5 文件
要使用这个脚本，需要先使用TCGA下载器（gdc-client）下载数据，还需要对应manifest文件读取
@author: Zhengwu Long <longzhengwu2236@gmail.com>
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.model_selection import train_test_split

sCol = ["read_count", "reads_per_million_miRNA_mapped", "cross-mapped"]
sKey  =  'miRNA_ID'
clin_item = ["case_submitter_id", "project_id", "ethnicity", "gender", "race", "vital_status", 
             "ajcc_pathologic_m", "ajcc_pathologic_n", "ajcc_pathologic_stage",
             "ajcc_pathologic_t", "treatment_or_therapy", "treatment_type",
             "days_to_birth","year_of_birth", "age_at_diagnosis", "age_at_index",
             "year_of_diagnosis", "days_to_last_follow_up", "days_to_death",]

cancer_type = ['Noraml', 'BLCA', 'BRCA', 'COAD', 'HNSC', 'KIRC', 'LIHC', 'LUAD', 'LUSC', 'OV', 'THCA']

sur_type = ["case_submitter_id", "project_id", "isCancer", "vital_status", 
            "days_to_last_follow_up", "days_to_death", 
            "race","gender","age_at_index",
            "ajcc_pathologic_m","ajcc_pathologic_n",
            "ajcc_pathologic_stage","ajcc_pathologic_t",
            "treatment_or_therapy", "treatment_type"]
class MiRNA_Expr_Reader():
    def __init__(self):
        self.File_ID = None
        self.miRNA_ID= None
        self.arr_rc = None
        self.arr_rpm = None
        self.arr_cm = None
        self.labels = None
        self.clinInfo = None
        self.clinInfos = None
    
    def read_mani(self, pre_path, mani_txt):
        if not os.path.exists(pre_path):
            raise FileNotFoundError("%s not exists"%(pre_path))
        mani_tab = pd.read_table(os.path.join(pre_path, mani_txt))
        mani_tab["path"] = mani_tab.apply(lambda x: os.path.join(pre_path,x.iloc[0],x.iloc[1]), axis=1)
        
        tmp_tab = pd.read_table(mani_tab['path'].iloc[0])
        self.miRNA_ID = tmp_tab["miRNA_ID"]
        rr = mani_tab.shape[0]
        cc = tmp_tab.shape[0]
        
        rc_arr = np.zeros((rr, cc))
        rpm_arr = rc_arr.copy()
        cm_arr = rc_arr.copy()
    
        for i in range(rr):
            tmp_path = mani_tab['path'].iloc[i]
            tmp_tab = pd.read_table(tmp_path)
            rc_arr[i,:] = np.array(tmp_tab[sCol[0]])
            rpm_arr[i,:] = np.array(tmp_tab[sCol[1]])
            cm_arr[i,:] = np.apply_along_axis(lambda x : x == 'Y', 0, np.array(tmp_tab[sCol[2]])).astype('int')
        

        self.File_ID = mani_tab['id']
        self.arr_rc = rc_arr
        self.arr_rpm = rpm_arr
        self.arr_cm = cm_arr
        
    def read_label(self, label_path, sep = '\t'):
        if (self.File_ID is None):
            raise AttributeError("Please read manifest file before")
        mini_labs = pd.read_csv(label_path, sep = sep)
        tmp_keys = pd.DataFrame({'File ID':self.File_ID})
        mini_labs = pd.merge(tmp_keys, mini_labs, on="File ID")
        sample_ID = mini_labs['Sample ID']
        proj = mini_labs['Project ID']
        proj = proj.apply(lambda x: x[5:])
        isCancer = sample_ID.apply(lambda x: x.split(sep="-")[-1][0] == "0").astype(int)
        self.labels = pd.DataFrame({'Sample ID': sample_ID, "Project ID": proj, "isCancer": isCancer})
        self.labels["case_submitter_id"] = self.labels["Sample ID"].apply(lambda x: x[:12])
        
        dict_cancer_type = {i: j for i, j in zip(cancer_type, range(len(cancer_type)))}
        self.labels["Project ID"] = self.labels["Project ID"].map(dict_cancer_type)
        
    def read_clin(self, clin_path, item = [],sep = '\t'):
        if (self.labels is None):
            raise AttributeError("Please read label file before")
        tmp_csv = pd.read_csv(clin_path, sep = '\t', na_values= "'--")
        tmp_csv = tmp_csv.iloc[range(0, tmp_csv.shape[0], 2), ]
        tmp_csv.index = range(len(tmp_csv))
        clin_data = tmp_csv[clin_item]
        clin_data.loc[:,"days_to_birth"]  = -clin_data.loc[:,"days_to_birth"]
        
        #用均值填充连续值中的nan
        for i in range(4, 0, -1):
            clin_data.iloc[pd.isna(clin_data.iloc[:,-i]),-i] = clin_data.iloc[:,-i].mean()
        #数字替换AJJC癌症分期，忽视ABC
        dict_ajjc_stage = {'Stage I':1, 'Stage III':3, 'Stage II':2, 'Stage IVC':4, 'Stage IVA':4,
               'Stage IIA':2, 'Stage IIB':2, 'Stage IIIA':3, 'Stage IIIB':3, 'Stage IIIC':3,
               'Stage IV':4, 'Stage IA':1, 'Stage X':-1, 'Stage IB':1, 'Stage IVB':4,
               'Stage IIC':2}
        clin_data["ajcc_pathologic_stage_n"] = clin_data["ajcc_pathologic_stage"].map(dict_ajjc_stage)
        clin_data.loc[pd.isna(clin_data["ajcc_pathologic_stage_n"]) ,"ajcc_pathologic_stage_n"] = -1
        
        #ethnicity gender rice alive treatment
        dict_ethn_isHispanicLatino = {'not reported':-1, 'not hispanic or latino':0, 'hispanic or latino':1}
        dict_gender_isMale = {'female':0, 'male':1}
        dict_race = {'not reported':-1, 'white':0, 'asian':1, 'black or african american':2,
               'american indian or alaska native':3,'native hawaiian or other pacific islander':4}
        dict_isAlive = {'Alive':1, 'Dead':0, 'Not Reported':-1}
        dict_isTreatment = {'yes':1, 'no':0, 'not reported':-1}
    
        clin_data["ethn_isHispanic_And_Latino"] = clin_data["ethnicity"].map(dict_ethn_isHispanicLatino) 
        clin_data["gender_isMale"] = clin_data["gender"].map(dict_gender_isMale) 
        clin_data["race_n"] = clin_data["race"].map(dict_race) 
        clin_data["vital_isAlive"] = clin_data["vital_status"].map(dict_isAlive)
        clin_data["isTreatment"] = clin_data["treatment_or_therapy"].map(dict_isTreatment)
        
        self.clinInfo = clin_data
    
    def read_clin_v2(self, clin_path, item = [],sep = '\t'):
        if (self.labels is None):
            raise AttributeError("Please read label file before")
        tmp_csv = pd.read_csv(clin_path, sep = '\t', na_values= "'--")
        tmp_csv = tmp_csv.iloc[range(0, tmp_csv.shape[0], 2), ]
        tmp_csv.index = range(len(tmp_csv))
        clin_data = tmp_csv[clin_item]
        clin_data.loc[:,"days_to_birth"]  = -clin_data.loc[:,"days_to_birth"]
        

        self.clinInfo = clin_data
        
        
    def data_preprocessing(self, idxClin = None):
        #merge labels and clinInfo, using set
        #delete sum=0 miRNA
        if (self.labels is None) or (self.clinInfo is None):
            raise AttributeError("Please read label file before")
        diffSet = set(self.labels["case_submitter_id"]) - set(self.clinInfo["case_submitter_id"])
        idxSample = self.labels["case_submitter_id"].apply(lambda x: x not in diffSet)
        self.arr_cm = self.arr_cm[idxSample,:]
        self.arr_rc = self.arr_rc[idxSample,:]
        self.arr_rpm = self.arr_rpm[idxSample,:]
        
        idxMiRNA = self.arr_rc.sum(axis=0) > 0
        self.arr_cm = self.arr_cm[:, idxMiRNA]
        self.arr_rc = self.arr_rc[:, idxMiRNA]
        self.arr_rpm = self.arr_rpm[:, idxMiRNA]
        self.miRNA_ID = self.miRNA_ID.iloc[idxMiRNA]
        
        #merge
        tmp_rank = "tmp_rank"
        idxClin_ = ['case_submitter_id', 'Project ID', 'isCancer', 'age_at_diagnosis', 'ajcc_pathologic_stage_n', 
                   'ethn_isHispanic_And_Latino','gender_isMale', 'race_n', 'vital_isAlive', 'isTreatment']
        if idxClin is None:
            idxClin = idxClin_

        self.labels["tmp_rank"] = range(self.labels.shape[0])
        clinInfos = pd.merge(self.labels, self.clinInfo, on = "case_submitter_id").loc[:, idxClin+[tmp_rank]]
        clinInfos = clinInfos.sort_values(tmp_rank)
        clinInfos.index = range(clinInfos.shape[0])
        clinInfos = clinInfos.drop(tmp_rank, axis=1)
        self.clinInfos = clinInfos
        
    
    def save_h5(self, h5_path, reWrite=True):
        # if (os.path.exists(h5_path)):
        #     raise FileExistsError("%s is existed" %(h5_path))
        miRNA_ID = [i.encode("utf-8") for i in self.miRNA_ID]
        file = h5py.File(h5_path, mode='w')
        file.create_dataset("miRNA_ID", data=miRNA_ID)
        file.create_dataset("arr_rc", data=self.arr_rc)
        file.create_dataset("arr_rpm", data=self.arr_rpm)
        file.create_dataset("arr_cm", data=self.arr_cm)
        self.clinInfos.to_hdf(h5_path, "clinInfos", append=True)
        file.close()
        
    def load_h5(self, h5_path):
        if not (os.path.exists(h5_path)):
            raise FileExistsError("%s not exist" %(h5_path))
        file = h5py.File(h5_path, mode='r')
        miRNA_ID = file['miRNA_ID'][:]
        self.arr_rc = file['arr_rc'][:]
        self.arr_rpm = file['arr_rpm'][:]
        self.arr_cm = file['arr_cm'][:]
        self.miRNA_ID = list(map(lambda x: str(x, encoding='utf-8'), miRNA_ID))
        self.clinInfos = pd.read_hdf(h5_path, "clinInfos")
        file.close()
        return self
    def Load_One_Cancer_lab(self, Cancer_n, test_size=0.3, random_state=42):
        idx = self.clinInfos.loc[:, "Project ID"] == Cancer_n
        X = self.arr_rc[idx]
        y = self.clinInfos.loc[idx, "isCancer"]
        
        y = np.array(y)
        if test_size is None:
            return X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=True)
            return X_train, X_test, y_train, y_test
        
    def Load_One_Cancer_mir_clin(self, Cancer_n="TCGA-LIHC", test_size=0.3, random_state=42):
        idx = self.clinInfos.loc[:, "project_id"] == Cancer_n
        X = self.arr_rc[idx]
        tmp = list(self.clinInfos.columns)
        tmp.remove("isCancer")
        y = self.clinInfos.loc[idx, "isCancer"]
        cli = self.clinInfos.loc[idx, tmp]
        y = np.array(y)
        if test_size is None:
            return X, y
        else:
            X_train, X_test, y_train, y_test, clin_train, clin_test = train_test_split(
                X, y, cli, test_size=test_size, random_state=random_state, shuffle=True)
            # clin_train = clin_train.reset_index(drop=True)
            # clin_test = clin_test.reset_index(drop=True)
            return X_train, X_test, y_train, y_test, clin_train, clin_test
        
        
    def Load_N0_Cancer_lab(self, Cancer_lst, test_size=0.3, random_state=42):
        idx = self.clinInfos.loc[:, "Project ID"] == Cancer_lst[0]
        for i in range(1, len(Cancer_lst)):
            # print(i)
            idx = (idx | (self.clinInfos.loc[:, "Project ID"] == Cancer_lst[i]))
        
        X = self.arr_rc[idx]
        y = self.clinInfos.loc[idx, "Project ID"]
        y1 = self.clinInfos.loc[idx, "isCancer"]
        y[y1 == 0] = 0 
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True)
        return X_train, X_test, y_train, y_test
        
    def Load_Ns_Cancer_lab(self, Cancer_lst, test_size=0.3, random_state=42):
        idx = self.clinInfos.loc[:, "Project ID"] == Cancer_lst[0]
        for i in range(1, len(Cancer_lst)):
            # print(i)
            idx = (idx | (self.clinInfos.loc[:, "Project ID"] == Cancer_lst[i]))
        
        X = self.arr_rc[idx]
        y = self.clinInfos.loc[idx, "Project ID"]
        y1 = self.clinInfos.loc[idx, "isCancer"]
        y[y1 == 0] += len(Cancer_lst)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True)
        return X_train, X_test, y_train, y_test
        
def Load_miRNA_h5():
    return MiRNA_Expr_Reader().load_h5("../Source_Data/miRNA_Data_h5/cancer10_miRNA.h5")
def main():
    a = MiRNA_Expr_Reader()
    a.read_mani("../Source_Data/miRNA_expr", "gdc_manifest_all.txt")
    a.read_label("../Source_Data/gdc_sample_sheet.2022-05-16.tsv", sep = '\t')
    a.read_clin("../Source_Data/clinical.tsv", item=clin_item)
    a.data_preprocessing()
    a.save_h5("../Source_Data/miRNA_Data_h5/cancer10_miRNA.h5")

    b = MiRNA_Expr_Reader()
    b.load_h5("../Source_Data/miRNA_Data_h5/cancer10_miRNA.h5")
    return b
def main2():
    a = MiRNA_Expr_Reader()
    a.read_mani("../Source_Data/miRNA_expr", "gdc_manifest_all.txt")
    a.read_label("../Source_Data/gdc_sample_sheet.2022-05-16.tsv", sep = '\t')
    a.read_clin_v2("../Source_Data/clinical.tsv", item=sur_type)
    a.data_preprocessing(sur_type)
    a.save_h5("../Source_Data/miRNA_Data_h5/cancer10_miRNA_v2.h5")

    b = MiRNA_Expr_Reader()
    b.load_h5("../Source_Data/miRNA_Data_h5/cancer10_miRNA_v2.h5")
    return b
if __name__ == "__main__":
    a = main2()



