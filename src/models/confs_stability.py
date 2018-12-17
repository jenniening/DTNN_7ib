### script used to compute stability for platinum dataset ###

import pandas as pd
import numpy as np
from ase.units import Hartree, eV, Bohr, Ang, mol, kcal

df = pd.read_csv("../../reports/result_data/platinummmff_test_result.csv")
df['target'] = df['target'].apply(lambda x : float(x))
df['tlmmff'] = df['tlmmff'].apply(lambda x : float(x))
df['dtnn7id'] = df['dtnn7id'].apply(lambda x : float(x))
df['tlconfmmff'] = df['tlconfmmff'].apply(lambda x : float(x))
df["target"] = df["target"]/(kcal/mol)
df['tlmmff'] = df["tlmmff"]/(kcal/mol)
df['dtnn7id'] = df['dtnn7id'] /(kcal/mol)
df['tlconfmmff'] = df['tlconfmmff'] / (kcal/mol)
df_total = df

df_stable = df_total.groupby("molecule_id").agg({"target":np.min,"tlmmff":np.min,"dtnn7id":np.min,"tlconfmmff":np.min, "mmffenergy":np.min})
df_stable.reset_index(inplace = True)

df_total_new = pd.merge(df_total,df_stable, on = ["molecule_id"])

df_total_new["diff_target"] = df_total_new["target_x"]  - df_total_new["target_y"]
df_total_new["diff_tlmmff"] = df_total_new["tlmmff_x"]  -df_total_new["tlmmff_y"]
df_total_new["diff_dtnn7id"] = df_total_new["dtnn7id_x"]  -df_total_new["dtnn7id_y"]
df_total_new["diff_tlconfmmff"] = df_total_new["tlconfmmff_x"]  - df_total_new["tlconfmmff_y"]
df_total_new["diff_MMFF"] = df_total_new["mmffenergy_x"]  - df_total_new["mmffenergy_y"]

def cal_MAE(target,predict):
    MAE = np.abs(target - predict).mean()
    
    return MAE

def get_MAE(df_sub):

    model_name = ["diff_MMFF","diff_tlmmff","diff_dtnn7id","diff_tlconfmmff"]
    MAES = {name:[] for name in model_name}
    remove_idx = []
    for i in set(df_sub["molecule_id"]):
        df_tmp = df_sub[df_sub["molecule_id"] == i]
        if df_tmp.shape[0] != 1:
            for name in model_name:
                MAE = cal_MAE(df_tmp["diff_target"], df_tmp[name])
                MAES[name].append(MAE)
        else:
            remove_idx.append(i)
    if df_sub["molecule_id"].shape[0] == 3667:
        for i in set(df_sub["molecule_id"]):
            df_tmp = df_sub[df_sub["molecule_id"] == i]
            if df_tmp.shape[0] != 1:
                MAE = cal_MAE(df_tmp["diff_target"], df_tmp[name])
                out_R_MAEvalue.write(str(i) + "," + str(MAE) + "\n")
            else:
                out_R_MAEvalue.write(str(i) + ",None\n")

    MAES_mean = [str(round(np.mean(value),2)) for value in MAES.values()]
    
    return remove_idx, MAES_mean


def cal_RMSE(target,predict):
    RMSE = np.sqrt(((target-predict)**2).mean())
    return RMSE


def get_RMSE(df_sub):

    RMSES = {name:[] for name in model_name}
    remove_idx = []
    for i in set(df_sub["molecule_id"]):
        df_tmp = df_sub[df_sub["molecule_id"] == i]
        if df_tmp.shape[0] != 1:
            for name in model_name:
                RMSE = cal_RMSE(df_tmp["diff_target"], df_tmp[name])
                RMSES[name].append(RMSE)
        else:
            remove_idx.append(i)

    RMSES_mean = [str(round(np.mean(value),2)) for value in RMSES.values()]
    
    return RMSES_mean


def get_SR(remove_list,df_sub):
    SR = {name:[] for name in model_name}
    lowest_index = df_sub[(df_sub["diff_target"] == 0 )&(~df_sub["molecule_id"].isin(remove_idx))]["index"]
    n_total = len(set(df_sub["molecule_id"]))
    for name in model_name:
        df_tmp = df_sub.where(df_sub[df_sub[name] == 0]["index"].isin(lowest_index))
        df_tmp.dropna(inplace = True)
        n_sr = df_tmp.shape[0]
        sr = round(float(n_sr)/float(n_total),2)
        SR[name] = str(sr)
    SR = [sr for sr in SR.values()]
    return SR

def get_absolute_MAE(df):
    MAES = {name:0 for name in model_name_abolute}
    for name in model_name_abolute:
        MAE = cal_MAE(df["target_x"], df[name])
        MAES[name] = MAE

    if df["molecule_id"].shape[0] == 3667:
        for i in set(df["molecule_id"]):
            df_tmp = df_sub[df["molecule_id"] == i]
            MAE = cal_MAE(df_tmp["target_x"], df_tmp[name])
            out_A_MAEvalue.write(str(i) + "," + str(MAE) + "\n")


    MAES_mean = [str(round(value,2)) for value in MAES.values()]

    return MAES_mean

def get_absolute_RMSE(df):
    RMSES = {name:0 for name in model_name_abolute}
    for name in model_name_abolute:
        RMSE = cal_RMSE(df["target_x"], df[name])
        RMSES[name] = RMSE

    RMSES_mean = [str(round(value,2)) for value in RMSES.values()]

    return RMSES_mean





outfile = open("../../reports/result_data/Platinum_performance_total.csv","w")
out_A_MAEvalue = open("../../reports/result_data/Platinum_AMAE_tlconfmmff_total.csv","w")
out_R_MAEvalue = open("../../reports/result_data/Platinum_RMAE_tlconfmmfftotal.csv","w")
label = ["MMFF","tlmmff","dtnn7id","tlconfmmff"]
model_name = ["diff_MMFF","diff_tlmmff","diff_dtnn7id","diff_tlconfmmff"]
model_name_abolute = ["mmffenergy_x","tlmmff_x","dtnn7id_x","tlconfmmff_x"]

outfile.write("index,metrics," + ",".join(label) + "\n")
for i in ["10","11","12","total"]:
    
    if i == "10":
        df_sub = df_total_new[(df_total_new["molecule_id"] <=25 )]
    elif i == "11":
        df_sub = df_total_new[(df_total_new["molecule_id"] <=52) & (df_total_new["molecule_id"] >= 26)]
    elif i == "12":
        df_sub = df_total_new[(df_total_new["molecule_id"] <=74) & (df_total_new["molecule_id"] >= 53)]
    elif i == "total":
        df_sub = df_total_new
        print(df_sub.shape)
        #print(df_sub.shape[0])
    remove_idx,MAES_mean = get_MAE(df_sub)
    RMSES_mean = get_RMSE(df_sub)
    print(remove_idx)
    SR = get_SR(remove_idx,df_sub)
    print(len(remove_idx))
    MAES_mean_ab = get_absolute_MAE(df_sub)
    RMSES_mean_ab = get_absolute_RMSE(df_sub)
    outfile.write(i + ",A_MAE," + ",".join(MAES_mean_ab) + "\n")
    outfile.write(i + ",A_RMSE," + ",".join(RMSES_mean_ab) + "\n")
    outfile.write(i + ",R_MAE," + ",".join(MAES_mean) + "\n")
    outfile.write(i + ",R_RMSE," + ",".join(RMSES_mean) + "\n")
    outfile.write(i + ",R_SR," + ",".join(SR) + "\n")
out_A_MAEvalue.close()
out_R_MAEvalue.close()
outfile.close()